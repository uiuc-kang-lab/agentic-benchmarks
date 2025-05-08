#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------
// CUDA kernel that processes a chunk of the batch (from start_n, for N_chunk samples)
// Implements 1D convolution with optional bias. Designed to be launched on a
// specific CUDA stream to overlap computation with asynchronous memory transfers.
// ---------------------------------------------------------------------
__global__ void conv1d_forward_kernel_stream(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr, // may be null
    float* __restrict__ y,
    int start_n,   // starting batch index for this chunk
    int N_chunk,   // number of samples in this chunk
    int C_in,
    int L_in,
    int C_out,
    int K,
    int stride,
    int padding,
    int dilation,
    int groups,
    int L_out
) {
    // Improved thread mapping: each block handles an output channel and a tile of output positions
    int out_ch = blockIdx.x;                     // output channel index
    int out_pos = blockIdx.y * blockDim.x + threadIdx.x; // output position within L_out
    int n_local = blockIdx.z;                      // local batch index within this chunk

    if (out_pos >= L_out || n_local >= N_chunk)
        return;

    int n = start_n + n_local; // global batch index

    // Determine channel grouping
    int group_size_out = C_out / groups;
    int group_size_in  = C_in  / groups;
    int group_idx = out_ch / group_size_out;

    float result = 0.0f;

    // Compute convolution for this output element
    for (int local_in_ch = 0; local_in_ch < group_size_in; local_in_ch++) {
        int in_ch = group_idx * group_size_in + local_in_ch;
        for (int k = 0; k < K; k++) {
            int in_pos = out_pos * stride + k * dilation - padding;
            if (in_pos >= 0 && in_pos < L_in) {
                float x_val = x[n * (C_in * L_in) + in_ch * L_in + in_pos];
                float w_val = w[out_ch * (group_size_in * K) + local_in_ch * K + k];
                result += x_val * w_val;
            }
        }
    }

    if (bias_ptr) {
        result += bias_ptr[out_ch];
    }

    y[n * (C_out * L_out) + out_ch * L_out + out_pos] = result;
}

// ---------------------------------------------------------------------
// Host implementation that splits the batch dimension into chunks and
// launches the convolution kernel on multiple CUDA streams to overlap
// computation with potential asynchronous memory transfers.
// ---------------------------------------------------------------------
at::Tensor conv1d_forward_impl(
    const at::Tensor& x,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");

    // x: [N, C_in, L_in]
    auto x_sizes = x.sizes();
    int64_t N_total = x_sizes[0];
    int64_t C_in = x_sizes[1];
    int64_t L_in = x_sizes[2];

    // weight: [C_out, C_in/groups, K]
    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0];
    int64_t K = w_sizes[2];

    // Calculate output length
    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    // Create output tensor
    auto y = torch::empty({N_total, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Determine the number of streams and batch chunk size
    int num_streams = (N_total < 4) ? N_total : 4; // use up to 4 streams
    int chunk_size = (N_total + num_streams - 1) / num_streams;  // ceiling division

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    // Set kernel launch parameters
    // We'll use blockDim.x = 256 threads to cover the L_out dimension
    dim3 blockSize(256);
    dim3 gridSize;
    gridSize.x = C_out;                       // one block per output channel
    gridSize.y = (L_out + blockSize.x - 1) / blockSize.x; // blocks to cover output positions
    
    // Launch kernels for each batch chunk on separate streams
    for (int i = 0; i < num_streams; i++) {
        int start_n = i * chunk_size;
        if (start_n >= N_total) break; // no more batches
        int current_chunk = std::min(chunk_size, (int)(N_total - start_n));

        // Adjust gridSize.z to process the batch in this chunk
        dim3 grid = gridSize;
        grid.z = current_chunk; // one block per sample in the chunk

        conv1d_forward_kernel_stream<<<grid, blockSize, 0, streams[i]>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            y.data_ptr<float>(),
            start_n,
            current_chunk,
            (int)C_in,
            (int)L_in,
            (int)C_out,
            (int)K,
            (int)stride,
            (int)padding,
            (int)dilation,
            (int)groups,
            (int)L_out
        );

        // Check for launch errors asynchronously
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_stream launch failed: ", cudaGetErrorString(err));
    }

    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return y;
}

// ---------------------------------------------------------------------
// Pybind11 binding: exposing the convolution forward function with optional bias
// and overlapping computation with memory transfers using CUDA streams
// ---------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        [](at::Tensor x,
           at::Tensor weight,
           py::object bias_obj,
           int64_t stride,
           int64_t padding,
           int64_t dilation,
           int64_t groups) {
            c10::optional<at::Tensor> bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<at::Tensor>();
            }
            return conv1d_forward_impl(x, weight, bias, stride, padding, dilation, groups);
        },
        "1D Convolution forward (CUDA) with stream pipelining to overlap computation with memory transfers"
    );
}
