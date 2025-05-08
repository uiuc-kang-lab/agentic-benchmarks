#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel leverages shared memory to load a contiguous chunk of the input that is used
// by a block of output elements from a single (batch, channel) pair. Threads in a block load
// the required region into shared memory, and then each thread computes the max over its
// pooling window using the fast shared memory, reducing global memory accesses.

__global__ void max_pool1d_kernel_shared(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int num_channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const bool return_indices
) {
    // Declare dynamic shared memory for the input tile
    extern __shared__ float sdata[];

    // Each block handles a tile of contiguous output elements for one (batch, channel) pair.
    // Calculate the starting output index for this block (tile).
    int tile_start = blockIdx.x * blockDim.x;  // Starting output index for this tile
    int tid = threadIdx.x;                      // Thread id within the block

    // Decode batch and channel from the second grid dimension
    int linear_idx = blockIdx.y;                // linear index for (batch, channel)
    int b = linear_idx / num_channels;
    int c = linear_idx % num_channels;

    // Compute the starting global input index corresponding to the first element of the tile
    // For an output element o, the input window starts at: o * stride - padding.
    // Thus, for the first output in the tile (tile_start), the starting global input index is:
    int in_start = tile_start * stride - padding;

    // Compute the size of the shared memory region needed.
    // The last thread in the block corresponds to output index (tile_start + blockDim.x - 1).
    // Its input window extends to: (tile_start + blockDim.x - 1)*stride - padding + (kernel_size - 1)*dilation.
    // Therefore, the total number of elements to load is:
    int shared_size = (blockDim.x - 1) * stride + (kernel_size - 1) * dilation + 1;

    // Base pointer for the current (b, c) slice in the input
    int input_offset = b * num_channels * input_length + c * input_length;

    // Load the required input region into shared memory.
    // Each thread loads multiple elements in a loop to cover all elements of the shared region.
    for (int j = tid; j < shared_size; j += blockDim.x) {
        int global_index = in_start + j;
        if (global_index < 0 || global_index >= input_length)
            sdata[j] = -INFINITY;
        else
            sdata[j] = input[input_offset + global_index];
    }
    __syncthreads();

    // Each thread computes one output element from the tile if it exists
    int o_idx = tile_start + tid;  // Global output index for this thread
    if (o_idx < output_length) {
        // The local offset in shared memory for this output's window can be computed from:
        // (o_idx * stride - padding) - in_start = (tile_start + tid)*stride - padding - (tile_start*stride - padding) = tid * stride
        int local_offset = tid * stride;

        float max_val = -INFINITY;
        int max_idx = -1;
        // Loop over the pooling window
        #pragma unroll
        for (int k = 0; k < kernel_size; k++) {
            int shared_index = local_offset + k * dilation;
            float value = sdata[shared_index];
            if (value > max_val) {
                max_val = value;
                // Compute the global index in the input corresponding to this value
                max_idx = in_start + shared_index;
            }
        }

        // Write the result to the output tensor
        int out_offset = b * num_channels * output_length + c * output_length + o_idx;
        output[out_offset] = max_val;
        if (return_indices) {
            indices[out_offset] = max_idx;
        }
    }
}

// Host function that wraps the CUDA kernel launch
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

    // Compute output length based on pooling parameters
    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, options.dtype(torch::kInt64));
    }

    // Define the tile size (number of output elements processed per block in the x-dimension)
    const int tile_size = 128;
    // Grid configuration: each block handles a tile for one (batch, channel) pair
    const int grid_x = (output_length + tile_size - 1) / tile_size;
    const dim3 blocks(grid_x, batch_size * num_channels);
    const dim3 threads(tile_size);

    // Compute the required shared memory size per block
    int shared_elems = (tile_size - 1) * stride + (kernel_size - 1) * dilation + 1;
    size_t shared_mem_bytes = shared_elems * sizeof(float);

    max_pool1d_kernel_shared<<<blocks, threads, shared_mem_bytes>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        return_indices ? indices.data_ptr<int64_t>() : nullptr,
        batch_size,
        num_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation,
        return_indices
    );

    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MaxPool1D forward with shared memory optimization (CUDA)");
}
