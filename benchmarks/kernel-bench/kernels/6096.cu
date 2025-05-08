#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel description:
// This kernel uses shared memory tiling to load the required input patch for a block's pooling windows.
// Each block processes one (n, c) slice of the input and computes a tile of output elements.
// By loading a shared memory tile that covers the pooling windows (with appropriate padding handled during load),
// the inner loops for accumulation avoid conditional boundary checks, ensuring uniform control flow and minimizing warp divergence.


template <typename scalar_t>
__global__ void shared_tile_avg_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int N,
    const int C,
    const int H,
    const int W,
    const int outH,
    const int outW,
    const int kernel_size,
    const int stride,
    const int padding
) {
    // Define tile dimensions for output
    const int TILE_WIDTH  = 16;  // number of output elements per block in x-dimension
    const int TILE_HEIGHT = 16;  // number of output elements per block in y-dimension

    // Each block is responsible for one (n, c) slice
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc % C;

    // Compute global output coordinates for the current thread
    int out_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int out_y = blockIdx.y * TILE_HEIGHT + threadIdx.y;

    // Compute the top-left global coordinate of the input patch that this block needs
    // For output, the mapping is: in_coord = out_coord * stride - padding
    int in_x_origin = blockIdx.x * TILE_WIDTH * stride - padding;
    int in_y_origin = blockIdx.y * TILE_HEIGHT * stride - padding;

    // Shared memory tile dimensions: it must cover all pooling windows for the block
    // The rightmost output in the block is at (blockIdx.x * TILE_WIDTH + (TILE_WIDTH - 1)).
    // Its pooling window in x extends from (out_x*stride - padding) to (out_x*stride - padding + kernel_size - 1).
    // Therefore, the shared tile width is:
    int tile_in_width  = TILE_WIDTH * stride + kernel_size - stride;  // = TILE_WIDTH * stride + (kernel_size - stride)
    int tile_in_height = TILE_HEIGHT * stride + kernel_size - stride;

    // Allocate shared memory (dynamically allocated by the host kernel launch)
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    // Total number of elements in the shared memory tile
    int tile_size = tile_in_width * tile_in_height;
    int threads_per_block = blockDim.x * blockDim.y;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    // Cooperative loading of the input patch into shared memory.
    // Each thread loads multiple elements by striding through the shared memory tile.
    for (int idx = thread_id; idx < tile_size; idx += threads_per_block) {
        int tx = idx % tile_in_width;
        int ty = idx / tile_in_width;
        int gx = in_x_origin + tx;  // global x index
        int gy = in_y_origin + ty;  // global y index

        // Check bounds and load, if out-of-bounds, load zero (zero-padding for pooling)
        if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
            sdata[idx] = input[((n * C + c) * H + gy) * W + gx];
        } else {
            sdata[idx] = scalar_t(0);
        }
    }
    __syncthreads();

    // Only compute output if within bounds
    if (out_x < outW && out_y < outH) {
        // The pooling window in shared memory for the output element at (out_x, out_y) starts at:
        // global input index for output = out_coord * stride - padding. Relative to the shared memory,
        // the offset is: (out_x * stride - padding) - in_x_origin, which simplifies to threadIdx.x * stride.
        int s_x = threadIdx.x * stride;
        int s_y = threadIdx.y * stride;
        
        scalar_t sum = scalar_t(0);
        
        // Unrolled accumulation over the pooling window; no conditional branches are used here as sdata
        // already contains zero for out-of-bound values.
        #pragma unroll
        for (int ky = 0; ky < 32; ky++) { // use a fixed loop bound; we'll break if ky >= kernel_size
            if (ky >= kernel_size) break;
            int s_row = (s_y + ky) * tile_in_width + s_x;
            #pragma unroll
            for (int kx = 0; kx < 32; kx++) { // use fixed loop bound; break if kx >= kernel_size
                if (kx >= kernel_size) break;
                sum += sdata[s_row + kx];
            }
        }
        
        int out_idx = ((n * C + c) * outH + out_y) * outW + out_x;
        output[out_idx] = sum / static_cast<scalar_t>(kernel_size * kernel_size);
    }
}


// Host forward function

torch::Tensor shared_tile_avg_pool2d_forward(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding
) {
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor.");
    
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    
    // Calculate output dimensions
    const int outH = (H + 2 * padding - kernel_size) / stride + 1;
    const int outW = (W + 2 * padding - kernel_size) / stride + 1;
    
    auto x_contig = x.contiguous();
    auto output = torch::empty({N, C, outH, outW}, x.options());
    
    // Define block and grid dimensions
    const int TILE_WIDTH = 16;
    const int TILE_HEIGHT = 16;
    dim3 threads(TILE_WIDTH, TILE_HEIGHT);
    dim3 blocks((outW + TILE_WIDTH - 1) / TILE_WIDTH, (outH + TILE_HEIGHT - 1) / TILE_HEIGHT, N * C);
    
    // Compute shared memory tile dimensions
    int tile_in_width  = TILE_WIDTH * stride + kernel_size - stride;
    int tile_in_height = TILE_HEIGHT * stride + kernel_size - stride;
    size_t shared_mem_bytes = tile_in_width * tile_in_height * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "shared_tile_avg_pool2d_kernel", ([&] {
        shared_tile_avg_pool2d_kernel<scalar_t><<<blocks, threads, shared_mem_bytes>>>(
            x_contig.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C, H, W, outH, outW,
            kernel_size, stride, padding
        );
    }));
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_tile_avg_pool2d_forward, "Shared Tile 2D Average Pooling forward (CUDA)");
}
