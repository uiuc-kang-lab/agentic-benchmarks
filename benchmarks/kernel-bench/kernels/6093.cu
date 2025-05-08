#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses shared memory tiling to reduce redundant global memory accesses for 2D average pooling.
// Each block loads a tile of the input into shared memory. Only one __syncthreads() is used to ensure that
// all threads have loaded their portion of the tile before any thread computes its output. This minimizes
// synchronization overhead while ensuring correct shared memory consistency.

template <typename scalar_t>
__global__ void shared_tile_avg_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int H,
    const int W,
    const int outH,
    const int outW,
    const int kernel_size,
    const int stride,
    const int padding
) {
    // Each block works on one (n, c) channel; blockIdx.z represents the combined index for (n, c).
    int nc = blockIdx.z;

    // Compute output coordinate for this thread
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Compute the origin (top-left) of the input tile for this block
    // For output, the mapping is: global_input_x = out_x * stride - padding
    // For the block, tile origin is computed from the block's first output element.
    int tile_origin_x = blockIdx.x * blockDim.x * stride - padding;
    int tile_origin_y = blockIdx.y * blockDim.y * stride - padding;

    // Determine shared memory tile dimensions. Each block computes a tile covering a block of outputs.
    // The tile must be large enough to cover the pooling windows for all outputs in the block.
    // tile_width = blockDim.x * stride + (kernel_size - stride)
    // tile_height = blockDim.y * stride + (kernel_size - stride)
    int tile_width  = blockDim.x * stride + (kernel_size - stride);
    int tile_height = blockDim.y * stride + (kernel_size - stride);
    int tile_size = tile_width * tile_height;

    // Allocate shared memory (declared as extern). Only one __syncthreads() will be used after loading.
    extern __shared__ char smem[]; // raw shared memory
    scalar_t* shared_tile = reinterpret_cast<scalar_t*>(smem);

    // Each thread in the block loads part of the shared memory tile.
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y;

    // For the current (n, c) channel, compute the pointer offset.
    const scalar_t* input_ptr = input + nc * H * W;

    // Load the tile from global memory into shared memory.
    // Some threads may load more than one element if tile_size > block_threads.
    for (int idx = thread_id; idx < tile_size; idx += block_threads) {
        int local_x = idx % tile_width;
        int local_y = idx / tile_width;
        int global_x = tile_origin_x + local_x;
        int global_y = tile_origin_y + local_y;
        if (global_x >= 0 && global_x < W && global_y >= 0 && global_y < H) {
            shared_tile[idx] = input_ptr[global_y * W + global_x];
        } else {
            shared_tile[idx] = scalar_t(0);
        }
    }

    // Synchronize threads to ensure the shared tile is fully loaded.
    __syncthreads();

    // Only compute the output if within bounds
    if (out_x < outW && out_y < outH) {
        // The pooling window for this output in global coordinates:
        //   in_x = out_x * stride - padding, in_y = out_y * stride - padding.
        // In shared memory, the offset is relative to tile_origin.
        // Given our setup, shared memory coordinate for this output's pooling window is:
        //   shared_x = threadIdx.x * stride
        //   shared_y = threadIdx.y * stride
        int shared_x = threadIdx.x * stride;
        int shared_y = threadIdx.y * stride;

        scalar_t sum = scalar_t(0);
        // Sum over the pooling window
        #pragma unroll
        for (int i = 0; i < kernel_size; i++) {
            int sy = shared_y + i;
            #pragma unroll
            for (int j = 0; j < kernel_size; j++) {
                int sx = shared_x + j;
                sum += shared_tile[sy * tile_width + sx];
            }
        }

        // Compute the index for the output tensor. The output is stored as (NC, outH, outW).
        int out_idx = nc * (outH * outW) + out_y * outW + out_x;
        output[out_idx] = sum / static_cast<scalar_t>(kernel_size * kernel_size);
    }
}

// Host function launching the shared memory tiling kernel

torch::Tensor shared_tile_avg_pool2d_forward(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding
) {
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor.");

    // x has shape: (N, C, H, W); we combine N and C into one dimension for simplicity.
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int outH = (H + 2 * padding - kernel_size) / stride + 1;
    int outW = (W + 2 * padding - kernel_size) / stride + 1;
    int NC = N * C;

    // Create an output tensor with shape (NC, outH, outW)
    auto x_cont = x.contiguous();
    auto options = x.options();
    auto output = torch::empty({NC, outH, outW}, options);

    // Configure the grid and block dimensions.
    // We use a 2D block for output spatial dimensions and grid.z for the combined (n, c) dimension.
    // Adjust block dimensions as appropriate; here we use 16x16 threads per block.
    dim3 threads(16, 16);
    dim3 blocks(
        (outW + threads.x - 1) / threads.x,
        (outH + threads.y - 1) / threads.y,
        NC
    );

    // Launch the kernel using shared memory tiling.
    // Compute the required shared memory size: tile dimensions depend on block size, stride, and kernel size.
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "shared_tile_avg_pool2d_kernel", ([&] {
        int tile_width = threads.x * stride + (kernel_size - stride);
        int tile_height = threads.y * stride + (kernel_size - stride);
        size_t shared_mem_size = tile_width * tile_height * sizeof(scalar_t);
        
        shared_tile_avg_pool2d_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            x_cont.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            H, W,
            outH, outW,
            kernel_size,
            stride,
            padding
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    // Reshape output to (N, C, outH, outW)
    return output.view({N, C, outH, outW});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_tile_avg_pool2d_forward, "Shared Tile 2D Average Pooling forward using minimal __syncthreads__ (CUDA)");
}
