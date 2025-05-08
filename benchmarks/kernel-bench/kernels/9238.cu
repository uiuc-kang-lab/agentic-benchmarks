#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Computes the output length for ConvTranspose1D
inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// Kernel that computes a chunk (subset of batch) of the ConvTranspose1D operation
__global__ void conv_transpose1d_kernel_chunk(
    const float* __restrict__ x_ptr,
    const float* __restrict__ weight_ptr,
    const float* __restrict__ bias_ptr,  // can be nullptr if bias not provided
    float* __restrict__ output_ptr,
    int batch_offset,         // offset in the batch dimension
    int batch_chunk_size,     // number of samples in this chunk
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_total = batch_chunk_size * out_channels * output_length;
    if (idx >= chunk_total) return;

    int local_b = idx / (out_channels * output_length);
    int rem = idx % (out_channels * output_length);
    int oc = rem / output_length;
    int o = rem % output_length;

    // Global batch index
    int b = local_b + batch_offset;
    float sum = 0.0f;
    
    // Loop over kernel positions
    for (int k = 0; k < kernel_size; ++k) {
        int i_pos = o + padding - k * dilation;
        if (i_pos % stride != 0) continue;
        int i = i_pos / stride;
        if (i < 0 || i >= input_length) continue;
        
        // Accumulate over input channels
        for (int ic = 0; ic < in_channels; ++ic) {
            int x_idx = b * in_channels * input_length + ic * input_length + i;
            int weight_idx = ic * out_channels * kernel_size + oc * kernel_size + k;
            sum += x_ptr[x_idx] * weight_ptr[weight_idx];
        }
    }
    
    // Add bias if provided
    if (bias_ptr) {
        sum += bias_ptr[oc];
    }

    int out_idx = b * out_channels * output_length + oc * output_length + o;
    output_ptr[out_idx] = sum;
}

// Forward function that overlaps computation with memory transfers using CUDA streams
// It partitions the batch dimension into chunks, launches the kernel for each chunk on a separate stream,
// and asynchronously copies the computed output from device to pinned host memory.
// The final result is returned as a CPU tensor (with pinned memory), which can later be used as needed.

torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, in_channels, input_length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");

    x = x.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_contig;
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_contig = bias->contiguous();
        TORCH_CHECK(bias_contig.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_contig.dim() == 1, "bias must be 1D");
        bias_ptr = bias_contig.data_ptr<float>();
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "weight's in_channels must match x's in_channels");
    if (bias.has_value()) {
        TORCH_CHECK(bias_contig.size(0) == out_channels, "bias size must match out_channels");
    }

    int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);

    // Allocate device output tensor without initialization to avoid extra zeroing overhead
    auto d_output = torch::empty({batch_size, out_channels, output_length}, x.options());

    // Allocate pinned host memory for output to enable asynchronous copy overlapping
    auto h_options = torch::TensorOptions().dtype(x.dtype()).device(torch::kCPU).pinned_memory(true);
    auto h_output = torch::empty({batch_size, out_channels, output_length}, h_options);

    // Create multiple CUDA streams to pipeline kernel execution and memory transfers
    int nstreams = 4;
    int chunk_size = (batch_size + nstreams - 1) / nstreams; // ceiling division to partition batch
    std::vector<cudaStream_t> streams(nstreams);
    for (int i = 0; i < nstreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int threads_per_block = 256;
    
    // Partition the batch dimension into chunks and process each chunk in a separate stream
    for (int i = 0; i < nstreams; i++) {
        int batch_offset = i * chunk_size;
        if (batch_offset >= batch_size) break;
        int current_chunk = std::min(chunk_size, batch_size - batch_offset);
        int num_elements = current_chunk * out_channels * output_length;
        int blocks = (num_elements + threads_per_block - 1) / threads_per_block;

        // Launch kernel for current batch chunk on stream[i]
        conv_transpose1d_kernel_chunk<<<blocks, threads_per_block, 0, streams[i]>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            d_output.data_ptr<float>(),
            batch_offset,
            current_chunk,
            in_channels,
            out_channels,
            input_length,
            output_length,
            kernel_size,
            stride,
            padding,
            dilation
        );

        // Asynchronously copy the computed chunk from device to pinned host memory
        size_t bytes = current_chunk * out_channels * output_length * sizeof(float);
        float* src = d_output.data_ptr<float>() + batch_offset * out_channels * output_length;
        float* dst = h_output.data_ptr<float>() + batch_offset * out_channels * output_length;
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize all streams to ensure computation and transfers are complete
    for (int i = 0; i < nstreams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Return the output tensor from pinned host memory
    return h_output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "ConvTranspose1D forward with pipelined multistream overlap (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}
