#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// This kernel processes a chunk of the batch dimension starting from batch_offset with chunk_size rows.
__global__ void stream_max_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_offset,
    const int chunk_size,
    const int num_channels,
    const int input_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int output_length,
    const bool return_indices) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = chunk_size * num_channels * output_length;  // Number of elements in this chunk
    if (tid >= total) return;

    // Decode the flattened index relative to the chunk
    int local_b = tid / (num_channels * output_length);
    int rem = tid % (num_channels * output_length);
    int c = rem / output_length;
    int i = rem % output_length;
    int b = batch_offset + local_b;  // Global batch index

    int input_start = i * stride - padding;
    float max_val = -INFINITY;
    int max_idx = -1;

    int base_idx = b * num_channels * input_length + c * input_length;

    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        int pos = input_start + k * dilation;
        if (pos >= 0 && pos < input_length) {
            float val = input[base_idx + pos];
            if (val > max_val) {
                max_val = val;
                max_idx = pos;
            }
        }
    }

    int out_index = b * num_channels * output_length + c * output_length + i;
    output[out_index] = max_val;
    if (return_indices) {
        indices[out_index] = max_idx;
    }
}

// Forward function that splits the input batch into chunks and uses multiple CUDA streams
// to overlap kernel execution (computation) with memory operations, improving pipelining.
// Although the input is already on device, splitting the workload allows concurrent kernel
// execution and any associated asynchronous memory operations.

torch::Tensor forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices) {

    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    const int batch_size = x.size(0);
    const int num_channels = x.size(1);
    const int input_length = x.size(2);

    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, options.dtype(torch::kInt64));
    }

    // Decide on a chunk size for pipelining. This can be tuned based on workload and GPU.
    int chunk_size = 8;
    if (chunk_size > batch_size) {
        chunk_size = batch_size;
    }
    int num_chunks = (batch_size + chunk_size - 1) / chunk_size;

    // Create CUDA streams for overlapping kernel execution and memory operations
    std::vector<cudaStream_t> streams(num_chunks);
    for (int i = 0; i < num_chunks; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    const int threads_per_block = 256;
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int batch_offset = chunk * chunk_size;
        int current_chunk_size = std::min(chunk_size, batch_size - batch_offset);
        int total_elements = current_chunk_size * num_channels * output_length;
        int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        // Launch the kernel asynchronously on the corresponding stream
        stream_max_pool1d_kernel<<<num_blocks, threads_per_block, 0, streams[chunk]>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            return_indices ? indices.data_ptr<int64_t>() : nullptr,
            batch_offset,
            current_chunk_size,
            num_channels,
            input_length,
            kernel_size,
            stride,
            padding,
            dilation,
            output_length,
            return_indices
        );
    }

    // Synchronize all streams to ensure kernel execution and any memory operations complete
    for (int i = 0; i < num_chunks; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MaxPool1D forward with stream overlap (CUDA)");
}
