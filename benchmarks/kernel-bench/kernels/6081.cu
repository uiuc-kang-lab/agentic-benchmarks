#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses shared memory to load a tile of the input, reducing global memory accesses.
// Threads synchronize only once after loading the shared memory tile to ensure consistency.
// Each block handles a single (n, c) slice of the input tensor and computes a tile of the output.

template <typename scalar_t>
__global__ void shared_avg_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernel_size, int stride, int padding
) {
    // Determine which (n, c) slice this block is processing
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc % C;

    // Compute global output coordinates for this thread
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Determine the shared memory tile dimensions
    // Each block computes a tile of output of size (blockDim.x, blockDim.y).
    // The corresponding input tile dimensions are computed based on stride and kernel size.
    int tile_w = blockDim.x * stride + (kernel_size - stride);
    int tile_h = blockDim.y * stride + (kernel_size - stride);

    // Calculate the top-left corner of the input tile to load into shared memory
    int in_tile_x = blockIdx.x * blockDim.x * stride - padding;
    int in_tile_y = blockIdx.y * blockDim.y * stride - padding;

    // Allocate shared memory dynamically. Each block gets a tile of size tile_w x tile_h.
    extern __shared__ char shmem[];
    scalar_t* tile = reinterpret_cast<scalar_t*>(shmem);

    // Cooperative loading of the shared memory tile
    int num_tile_elements = tile_w * tile_h;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;
    for (int i = tid; i < num_tile_elements; i += total_threads) {
        int tile_x = i % tile_w;
        int tile_y = i / tile_w;
        int in_x = in_tile_x + tile_x;
        int in_y = in_tile_y + tile_y;
        if (in_x >= 0 && in_x < W && in_y >= 0 && in_y < H) {
            int input_idx = ((n * C + c) * H + in_y) * W + in_x;
            tile[i] = input[input_idx];
        } else {
            tile[i] = scalar_t(0);
        }
    }
    // Synchronize to ensure the shared memory tile is fully loaded
    __syncthreads();

    // Compute the output if within bounds
    if (out_x < outW && out_y < outH) {
        // Calculate local coordinates in the shared memory tile for the top-left corner of the pooling window
        // The mapping: global_input_x = out_x * stride - padding, and our tile starts at in_tile_x
        int local_x = (out_x * stride - padding) - in_tile_x;  // equals threadIdx.x * stride
        int local_y = (out_y * stride - padding) - in_tile_y;    // equals threadIdx.y * stride

        scalar_t sum = scalar_t(0);
        // Unroll the inner loops for the pooling window
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ky++) {
            int offset = (local_y + ky) * tile_w;
            #pragma unroll
            for (int kx = 0; kx < kernel_size; kx++) {
                sum += tile[offset + local_x + kx];
            }
        }
        int out_idx = ((n * C + c) * outH + out_y) * outW + out_x;
        output[out_idx] = sum / static_cast<scalar_t>(kernel_size * kernel_size);
    }
    // No extraneous synchronizations are used beyond the necessary __syncthreads() after loading shared memory.
}


torch::Tensor shared_avg_pool2d_forward(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding
) {
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor.");
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int outH = (H + 2 * padding - kernel_size) / stride + 1;
    int outW = (W + 2 * padding - kernel_size) / stride + 1;

    auto x_cont = x.contiguous();
    auto options = x.options();
    auto output = torch::empty({N, C, outH, outW}, options);

    // Define block and grid dimensions
    const int block_x = 16;
    const int block_y = 16;
    dim3 threads(block_x, block_y);
    dim3 blocks((outW + block_x - 1) / block_x, (outH + block_y - 1) / block_y, N * C);

    // Compute the dimensions of the shared memory tile
    int tile_w = block_x * stride + (kernel_size - stride);
    int tile_h = block_y * stride + (kernel_size - stride);

    // Launch the kernel with dynamic shared memory; the shared memory size depends on the tile dimensions and data type.
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "shared_avg_pool2d_kernel", ([&] {
        size_t shared_mem_size = tile_w * tile_h * sizeof(scalar_t);
        const scalar_t* input_data = x_cont.data_ptr<scalar_t>();
        scalar_t* output_data = output.data_ptr<scalar_t>();
        shared_avg_pool2d_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input_data,
            output_data,
            N, C, H, W,
            outH, outW,
            kernel_size, stride, padding
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_avg_pool2d_forward, "Shared Memory 2D Average Pooling forward (CUDA)");
}
