#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile dimensions for output block
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// Kernel: Each block processes a tile of the output for one (n, c) slice.
// It loads a corresponding region from the input to shared memory to reuse data between output computations.
// We use __syncthreads() only once, after the shared memory load, to ensure consistency.

template <typename scalar_t>
__global__ void avg_pool2d_shared_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int H, int W,       // Input dimensions
    int outH, int outW, // Output dimensions
    int kernel_size,
    int stride,
    int padding
) {
    // Each block processes one (n, c) channel.
    // gridDim.z should equal N*C, and we index input accordingly.
    int nc = blockIdx.z;  // linear index for batch and channel
    int stride_input = H * W;
    const scalar_t* input_nc = input + nc * stride_input;
    scalar_t* output_nc = output + nc * (outH * outW);

    // Determine the output tile this block is responsible for
    int out_row_base = blockIdx.y * TILE_HEIGHT;
    int out_col_base = blockIdx.x * TILE_WIDTH;

    // Compute the corresponding top-left coordinate in the input for this tile
    // This is where the shared memory tile begins
    int in_tile_row = out_row_base * stride - padding;
    int in_tile_col = out_col_base * stride - padding;

    // Determine the size of the shared memory tile.
    // For an output tile of size TILE_HEIGHT x TILE_WIDTH, the required input patch size is:
    // shared_height = (TILE_HEIGHT - 1) * stride + kernel_size
    // shared_width  = (TILE_WIDTH  - 1) * stride + kernel_size
    int shared_height = (TILE_HEIGHT - 1) * stride + kernel_size;
    int shared_width  = (TILE_WIDTH  - 1) * stride + kernel_size;

    extern __shared__ char smem[];
    scalar_t* shmem = reinterpret_cast<scalar_t*>(smem);

    // Load data into shared memory cooperatively
    int num_shared_elements = shared_height * shared_width;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y;
    for (int idx = tid; idx < num_shared_elements; idx += block_threads) {
        int sh_r = idx / shared_width;
        int sh_c = idx % shared_width;
        int global_r = in_tile_row + sh_r;
        int global_c = in_tile_col + sh_c;
        if (global_r >= 0 && global_r < H && global_c >= 0 && global_c < W) {
            shmem[idx] = input_nc[global_r * W + global_c];
        } else {
            shmem[idx] = static_cast<scalar_t>(0);
        }
    }

    // Synchronize to ensure the shared memory tile is fully loaded
    __syncthreads();

    // Each thread computes one output element in the tile
    int out_row = out_row_base + threadIdx.y;
    int out_col = out_col_base + threadIdx.x;
    if (out_row < outH && out_col < outW) {
        // Compute the top-left index in shared memory for this pooling window
        // Relative offset in shared memory: (out_row * stride - in_tile_row) simplifies to (threadIdx.y * stride)
        int sh_row = threadIdx.y * stride;
        int sh_col = threadIdx.x * stride;

        scalar_t sum_val = static_cast<scalar_t>(0);
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                int index_sh = (sh_row + i) * shared_width + (sh_col + j);
                sum_val += shmem[index_sh];
            }
        }
        output_nc[out_row * outW + out_col] = sum_val / static_cast<scalar_t>(kernel_size * kernel_size);
    }
}


torch::Tensor avg_pool2d_forward(
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
    auto out = torch::empty({N, C, outH, outW}, options);

    // Define block dimensions: each block covers a TILE_HEIGHT x TILE_WIDTH output tile for one (n, c)
    dim3 threads(TILE_WIDTH, TILE_HEIGHT);
    int grid_x = (outW + TILE_WIDTH - 1) / TILE_WIDTH;
    int grid_y = (outH + TILE_HEIGHT - 1) / TILE_HEIGHT;
    // Each (n, c) slice gets its own grid block along z
    dim3 blocks(grid_x, grid_y, N * C);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_shared_kernel", ([&] {
        size_t shared_height = (TILE_HEIGHT - 1) * stride + kernel_size;
        size_t shared_width = (TILE_WIDTH - 1) * stride + kernel_size;
        size_t shared_mem_size = shared_height * shared_width * sizeof(scalar_t);

        const scalar_t* input_data = x_cont.data_ptr<scalar_t>();
        scalar_t* output_data = out.data_ptr<scalar_t>();
        avg_pool2d_shared_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input_data,
            output_data,
            H, W,
            outH, outW,
            kernel_size, stride, padding
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool2d_forward, "Optimized 2D Average Pooling forward (CUDA) using shared memory with minimal synchronization");
}
