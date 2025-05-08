#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block size for output tiling along the output length dimension
#define BLOCK_SIZE 256

// This kernel uses shared memory to load a tile of the input for each (batch, channel) pair.
// This ensures that global memory accesses are coalesced when loading the tile, as threads in a warp
// load contiguous elements. Each block processes a tile of consecutive output elements for a given
// channel and batch index. The kernel then uses the shared tile to perform max pooling over the
// pooling window, with proper handling of boundary conditions.

__global__ void max_pool1d_shared_coalesced_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int num_channels,
    const int input_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int output_length) {

    // Grid mapping:
    //   blockIdx.z -> batch index
    //   blockIdx.y -> channel index
    //   blockIdx.x -> tile index along the output dimension
    
    int b = blockIdx.z;   // batch index
    int c = blockIdx.y;   // channel index
    int tile = blockIdx.x; // tile index along output dimension

    int tx = threadIdx.x; // thread index within the tile

    // Each block handles BLOCK_SIZE contiguous output positions for one (b, c) pair.
    int out_index = tile * blockDim.x + tx;  // global output index within this channel

    // Base pointer for the current (b, c) in the input tensor
    int base_offset = b * num_channels * input_length + c * input_length;

    // Calculate the starting global input index corresponding to the first output of this tile.
    // For the first output in the tile (output index = tile * BLOCK_SIZE), the corresponding
    // start position in the input is: (output_index * stride - padding).
    int tile_output_start = tile * blockDim.x;  
    int shared_input_start = tile_output_start * stride - padding;

    // Determine the size of the shared memory tile that needs to be loaded.
    // For BLOCK_SIZE outputs, the input span required is:
    //   shared_size = (BLOCK_SIZE - 1)*stride + (kernel_size - 1)*dilation + 1
    int shared_size = (blockDim.x - 1) * stride + (kernel_size - 1) * dilation + 1;

    // Declare dynamic shared memory
    extern __shared__ float shmem[];

    // Each thread loads one or more elements into shared memory so that global memory access
    // is coalesced. The loaded tile covers [shared_input_start, shared_input_start + shared_size).
    for (int i = tx; i < shared_size; i += blockDim.x) {
        int input_idx = shared_input_start + i;
        if (input_idx < 0 || input_idx >= input_length) {
            shmem[i] = -INFINITY;
        } else {
            shmem[i] = input[base_offset + input_idx];
        }
    }
    __syncthreads();

    // Only process if the global output index is within range
    if (out_index < output_length) {
        // The location in shared memory corresponding to the start of the pooling window for this output.
        // Note: Global pooling window start = out_index * stride - padding.
        // Relative to our shared memory tile starting at shared_input_start, the offset is:
        //   local_start = (out_index * stride - padding) - shared_input_start
        // Given shared_input_start = tile_output_start * stride - padding, this simplifies to:
        //   local_start = (out_index - tile_output_start) * stride = tx * stride
        int local_start = tx * stride;
        float max_val = -INFINITY;
        int max_k = 0; // pooling window offset index that produces the max value
        
        // Iterate over the pooling window using the dilation factor
        for (int k = 0; k < kernel_size; ++k) {
            int sh_idx = local_start + k * dilation;
            float val = shmem[sh_idx];
            if (val > max_val) {
                max_val = val;
                max_k = k;
            }
        }
        
        // Calculate the corresponding global index of the maximum value
        int global_max_index = out_index * stride - padding + max_k * dilation;

        // Write the result to the output tensor
        int output_offset = b * num_channels * output_length + c * output_length + out_index;
        output[output_offset] = max_val;
        if (indices != nullptr) {
            indices[output_offset] = global_max_index;
        }
    }
}

// Host function wrapping the kernel launch
torch::Tensor forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices) {

    TORCH_CHECK(x.dim() == 3, "Input must be 3D, got ", x.dim());
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    const int batch_size = x.size(0);
    const int num_channels = x.size(1);
    const int input_length = x.size(2);

    // Compute output length based on the pooling parameters
    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, options.dtype(torch::kInt64));
    }

    // Configure the kernel launch dimensions
    // Each block processes a tile of BLOCK_SIZE consecutive output elements for one (batch, channel) pair.
    int num_tiles = (output_length + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(num_tiles, num_channels, batch_size);

    // Shared memory size per block (in bytes)
    int shared_mem_size = ((BLOCK_SIZE - 1) * stride + (kernel_size - 1) * dilation + 1) * sizeof(float);

    max_pool1d_shared_coalesced_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        return_indices ? indices.data_ptr<int64_t>() : nullptr,
        batch_size,
        num_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        dilation,
        output_length);

    // Optional: synchronize and check for errors
    cudaDeviceSynchronize();

    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MaxPool1D forward with shared memory for coalesced access (CUDA)");
}
