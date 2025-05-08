#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses shared memory to load a tile of input data for a (batch, channel) pair. 
// It synchronizes threads only once after loading shared memory, ensuring data consistency while avoiding excessive __syncthreads() calls.

__global__ void max_pool1d_shared_sync_opt_kernel(
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
    // Allocate dynamic shared memory
    extern __shared__ float sdata[];

    // Each block processes a tile of contiguous output elements for a specific (batch, channel)
    int tile_start = blockIdx.x * blockDim.x;  // starting output index for this tile
    int tid = threadIdx.x;                      // local thread id

    // Decode (batch, channel) from the second grid dimension
    int linear_idx = blockIdx.y;               
    int b = linear_idx / num_channels;
    int c = linear_idx % num_channels;

    // Compute the base pointer for the input corresponding to the (b,c) slice
    int base_input = b * num_channels * input_length + c * input_length;

    // Compute the starting index in the input for the first pooling window in this tile
    int global_in_start = tile_start * stride - padding;

    // Compute required shared memory size for this tile
    // For tile size T: shared_size = (T - 1) * stride + (kernel_size - 1) * dilation + 1
    int shared_size = (blockDim.x - 1) * stride + (kernel_size - 1) * dilation + 1;

    // Load data into shared memory. Each thread loads multiple elements in strides of blockDim.x
    for (int j = tid; j < shared_size; j += blockDim.x) {
        int in_pos = global_in_start + j;
        if (in_pos < 0 || in_pos >= input_length)
            sdata[j] = -INFINITY;
        else
            sdata[j] = input[base_input + in_pos];
    }
    // Synchronize to ensure all shared memory loads are complete
    __syncthreads();

    // Each thread computes one output element from the tile
    int out_idx = tile_start + tid;
    if (out_idx < output_length) {
        // The local offset in shared memory for this output's pooling window is tid * stride
        int local_offset = tid * stride;
        float max_val = -INFINITY;
        int max_idx = -1;
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            int s_index = local_offset + k * dilation;
            float val = sdata[s_index];
            if (val > max_val) {
                max_val = val;
                // Map the shared memory index back to the global input index
                max_idx = global_in_start + s_index;
            }
        }
        // Compute global output index
        int global_out_idx = b * num_channels * output_length + c * output_length + out_idx;
        output[global_out_idx] = max_val;
        if (return_indices) {
            indices[global_out_idx] = max_idx;
        }
    }
}

// Host wrapper function

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
    
    // Compute output length
    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, options.dtype(torch::kInt64));
    }

    // Define tile size and grid dimensions
    const int tile_size = 128;  // number of outputs processed per block (x-dimension)
    dim3 blocks((output_length + tile_size - 1) / tile_size, batch_size * num_channels);
    dim3 threads(tile_size);

    // Calculate shared memory size in bytes
    int shared_mem_size = (tile_size - 1) * stride + (kernel_size - 1) * dilation + 1;
    shared_mem_size *= sizeof(float);

    max_pool1d_shared_sync_opt_kernel<<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &forward, "MaxPool1D forward with optimized shared memory synchronization (CUDA)");
}
