#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// This kernel uses a tiled approach for 1D average pooling. Each block processes a tile
// of contiguous output elements (num_outputs_per_block outputs computed by separate warps).
// The required input region for the tile is loaded into shared memory, and each warp
// cooperatively computes one output element by summing over its pooling window using
// warp-level reduction (__shfl_down_sync). This reduces global memory accesses and
// leverages fast shared and warp-level operations without changing precision.

__global__ void avg_pool1d_kernel(
    const float *input,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int num_outputs_per_block) {

    // Identify the batch and channel from block indices
    int batch = blockIdx.z;
    int channel = blockIdx.y;

    // Each block processes a tile of output elements along the length dimension
    int tile_start = blockIdx.x * num_outputs_per_block;

    // Within the block, each warp (of 32 threads) computes one output element
    int warp_id = threadIdx.x / WARP_SIZE;  // which warp within the block
    int lane = threadIdx.x % WARP_SIZE;      // lane id within the warp

    // Compute the global output index for this warp
    int global_o = tile_start + warp_id;

    // Determine the shared memory region to load the necessary input data for the entire tile
    // The pooling window for an output element starts at: out_index * stride - padding
    // For the tile, the first required input index is:
    int smem_start = tile_start * stride - padding;
    // For the last output in the tile, the pooling window covers up to:
    int last_output = tile_start + num_outputs_per_block - 1;
    int smem_end = last_output * stride - padding + kernel_size;
    int smem_size = smem_end - smem_start; // number of elements to load into shared memory

    // Compute pointers for the current channel and batch in global memory
    // x is of shape [batch_size, in_channels, input_length]
    // gridDim.y was set to in_channels, so we can use it to compute strides
    int in_channels = gridDim.y; // as configured
    const float *input_ptr = input + batch * (in_channels * input_length) + channel * input_length;
    float *output_ptr = output + batch * (in_channels * output_length) + channel * output_length;

    // Allocate shared memory (dynamically-sized)
    extern __shared__ float smem[];

    // Each thread in the block loads elements of the shared memory tile cooperatively
    for (int i = threadIdx.x; i < smem_size; i += blockDim.x) {
        int global_idx = smem_start + i;
        // Load from global memory if within bounds; else set to 0
        if (global_idx >= 0 && global_idx < input_length)
            smem[i] = input_ptr[global_idx];
        else
            smem[i] = 0.0f;
    }
    __syncthreads();

    // Only process if the computed output index is within the valid range
    if (global_o < output_length) {
        // Compute the start of the pooling window in global memory
        int pool_start_global = global_o * stride - padding;
        // Compute the corresponding index in shared memory
        int pool_start_local = pool_start_global - smem_start;

        float partial_sum = 0.0f;
        // Each lane computes a portion of the sum over the kernel window
        const float* window = smem + pool_start_local;
        for (int k = lane; k < kernel_size; k += WARP_SIZE) {
            partial_sum += window[k];
        }

        // Use warp-level reduction to accumulate the partial sums
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }

        // Lane 0 of each warp writes the final averaged result
        if (lane == 0) {
            output_ptr[global_o] = partial_sum / kernel_size;
        }
    }
}


torch::Tensor avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {
    
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid kernel parameters");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    // We'll have each warp (32 threads) compute one output element.
    // Using 256 threads per block gives num_outputs_per_block = 256 / 32 = 8 outputs computed per block.
    int threads_per_block = 256;
    int num_outputs_per_block = threads_per_block / WARP_SIZE; // 8 outputs per block

    // Grid dimensions:
    //   - grid.x: number of tiles to cover output_length
    //   - grid.y: in_channels
    //   - grid.z: batch_size
    dim3 threads(threads_per_block);
    dim3 grid((output_length + num_outputs_per_block - 1) / num_outputs_per_block, in_channels, batch_size);

    // Shared memory size required per block (in number of floats):
    // It is independent of tile start, computed as: (num_outputs_per_block - 1) * stride + kernel_size
    int smem_size = (num_outputs_per_block - 1) * stride + kernel_size;

    avg_pool1d_kernel<<<grid, threads, smem_size * sizeof(float)>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel_size,
        stride,
        padding,
        input_length,
        output_length,
        num_outputs_per_block
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward (CUDA) with shared memory and warp-level reduction");
}
