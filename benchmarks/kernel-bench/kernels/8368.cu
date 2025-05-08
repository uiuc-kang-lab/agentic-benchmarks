#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

namespace py = pybind11;

// -----------------------------------------------------
// Standard 1D convolution kernel (each thread computes one output element)
// This kernel is written to work on a sub-batch of the input.
// -----------------------------------------------------
__global__ void conv1d_forward_kernel_stream(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias, // can be null
    float* __restrict__ y,
    const int N,       // number of samples in this sub-batch
    const int C_in,
    const int L_in,
    const int C_out,
    const int K,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int L_out
) {
    // Each thread computes one output element (n, out_ch, out_pos) for the sub-batch
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * L_out;
    if (idx >= total) return;

    int out_pos = idx % L_out;
    int out_ch = (idx / L_out) % C_out;
    int n = idx / (L_out * C_out);

    int group_size_in = C_in / groups;
    int group_size_out = C_out / groups;
    int group_idx = out_ch / group_size_out;

    float val = 0.0f;
    for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
        int in_ch = group_idx * group_size_in + local_in_ch;
        for (int k = 0; k < K; ++k) {
            int in_pos = out_pos * stride + k * dilation - padding;
            if (in_pos >= 0 && in_pos < L_in) {
                float x_val = x[n * (C_in * L_in) + in_ch * L_in + in_pos];
                float w_val = w[out_ch * (group_size_in * K) + local_in_ch * K + k];
                val += x_val * w_val;
            }
        }
    }

    if (bias)
        val += bias[out_ch];

    y[n * (C_out * L_out) + out_ch * L_out + out_pos] = val;
}

// -----------------------------------------------------
// Convolution forward implementation using stream pipelining to overlap
// kernel execution with memory transfers. The input batch is partitioned
// into chunks, each processed on a separate CUDA stream. After kernel
// launch, the corresponding output chunk is asynchronously copied
// from device to pinned host memory. Finally, the output is copied back
// to device before returning.
// -----------------------------------------------------

at::Tensor conv1d_forward_impl_streamed(
    const at::Tensor& x,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    // Safety checks
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");

    // Get input dimensions
    const int64_t N = x.size(0);
    const int64_t C_in = x.size(1);
    const int64_t L_in = x.size(2);
    const int64_t C_out = weight.size(0);
    const int64_t K = weight.size(2);

    // Compute output length
    const int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    // Allocate device output tensor to store intermediate results
    auto y = torch::empty({N, C_out, L_out}, x.options());

    // Determine number of streams and partition the batch dimension
    int num_streams = (N < 4) ? N : 4;  // use up to 4 streams
    int chunk_size = (N + num_streams - 1) / num_streams;

    // Allocate pinned host memory to hold the full output (for overlapping transfers)
    size_t total_bytes = N * C_out * L_out * sizeof(float);
    float* host_output = nullptr;
    cudaError_t err = cudaHostAlloc((void**)&host_output, total_bytes, cudaHostAllocDefault);
    TORCH_CHECK(err == cudaSuccess, "cudaHostAlloc failed: ", cudaGetErrorString(err));

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaError_t s_err = cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        TORCH_CHECK(s_err == cudaSuccess, "cudaStreamCreateWithFlags failed: ", cudaGetErrorString(s_err));
    }

    // Launch kernels and asynchronously copy each chunk from device to host
    for (int chunk_idx = 0; chunk_idx < num_streams; chunk_idx++) {
        int start = chunk_idx * chunk_size;
        int current_chunk = std::min(chunk_size, (int)N - start);
        if (current_chunk <= 0) break;

        // Compute pointers for the current sub-batch
        const float* x_ptr = x.data_ptr<float>() + start * C_in * L_in;
        float* y_ptr = y.data_ptr<float>() + start * C_out * L_out;

        // Total number of threads for this sub-batch
        int total_threads_chunk = current_chunk * C_out * L_out;
        int blockSize = 32;
        int gridSize = (total_threads_chunk + blockSize - 1) / blockSize;

        // Launch the convolution kernel for this sub-batch on the corresponding stream
        conv1d_forward_kernel_stream<<<gridSize, blockSize, 0, streams[chunk_idx]>>>(
            x_ptr,
            weight.data_ptr<float>(),
            bias_opt.has_value() ? bias_opt.value().data_ptr<float>() : nullptr,
            y_ptr,
            current_chunk,
            C_in,
            L_in,
            C_out,
            (int)K,
            (int)stride,
            (int)padding,
            (int)dilation,
            (int)groups,
            (int)L_out
        );

        // Asynchronously copy the computed chunk (sub-batch) from device to pinned host memory
        size_t chunk_bytes = current_chunk * C_out * L_out * sizeof(float);
        err = cudaMemcpyAsync(host_output + start * C_out * L_out, y_ptr, chunk_bytes,
                              cudaMemcpyDeviceToHost, streams[chunk_idx]);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed: ", cudaGetErrorString(err));
    }

    // Synchronize all streams to ensure all kernel launches and copies are finished
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Optionally, copy the full output from host pinned memory back to a new device tensor
    // This final asynchronous copy overlaps with any potential subsequent work in a real pipeline.
    auto final_y = torch::empty({N, C_out, L_out}, x.options());
    err = cudaMemcpy(final_y.data_ptr<float>(), host_output, total_bytes, cudaMemcpyHostToDevice);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy (HostToDevice) failed: ", cudaGetErrorString(err));

    // Free the pinned host memory
    cudaFreeHost(host_output);

    return final_y;
}

// -----------------------------------------------------
// Pybind11 binding
// -----------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        [](at::Tensor x, at::Tensor weight, py::object bias,
           int64_t stride, int64_t padding, int64_t dilation, int64_t groups) {
            c10::optional<at::Tensor> bias_opt;
            if (!bias.is_none()) {
                bias_opt = bias.cast<at::Tensor>();
            }
            return conv1d_forward_impl_streamed(x, weight, bias_opt, stride, padding, dilation, groups);
        },
        "1D Convolution forward with overlapping computation and memory transfers using CUDA streams"
    );
}
