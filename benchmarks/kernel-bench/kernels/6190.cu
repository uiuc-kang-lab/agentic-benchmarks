#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel leverages shared memory to load a tile of the input that covers the pooling receptive field
// for a block of output elements. Each block processes a tile for one (n, c) plane. The shared memory tile
// reduces global memory latency by reusing data for neighboring output computations.

template <typename scalar_t>
__global__ void avg_pool2d_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernel_size, int stride, int padding
) {
    // Each block works on one (n, c) plane and a tile of output
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc % C;

    // Output tile base coordinates for this block
    int out_row_base = blockIdx.x * blockDim.y;
    int out_col_base = blockIdx.y * blockDim.x;

    // Corresponding top-left coordinate in the input for the shared memory tile
    int in_row_base = out_row_base * stride - padding;
    int in_col_base = out_col_base * stride - padding;

    // Dimensions of the shared memory tile needed
    // Each output pixel uses a kernel_size x kernel_size window. For a block of size (blockDim.y, blockDim.x),
    // the shared memory tile dimensions are computed as:
    // tileH = (blockDim.y - 1) * stride + kernel_size
    // tileW = (blockDim.x - 1) * stride + kernel_size
    int tileH = blockDim.y * stride + kernel_size - stride; // equivalent to (blockDim.y - 1)*stride + kernel_size
    int tileW = blockDim.x * stride + kernel_size - stride; // equivalent to (blockDim.x - 1)*stride + kernel_size

    // Allocate shared memory dynamically
    extern __shared__ char smem[];
    scalar_t* shmem = reinterpret_cast<scalar_t*>(smem);

    // Total number of elements in the shared tile
    int tile_size = tileH * tileW;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;

    // Cooperative loading: Each thread loads multiple elements of the tile if needed
    for (int idx = tid; idx < tile_size; idx += num_threads) {
        int sh_row = idx % tileH;
        int sh_col = idx % tileW;
        int in_row = in_row_base + sh_row;
        int in_col = in_col_base + sh_col;
        if (in_row >= 0 && in_row < H && in_col >= 0 && in_col < W) {
            shmem[idx] = input[((n * C + c) * H + in_row) * W + in_col];
        } else {
            shmem[idx] = scalar_t(0);
        }
    }
    __syncthreads();

    // Each thread computes one output element from its corresponding pooling window
    int out_row = out_row_base + threadIdx.y;
    int out_col = out_col_base + threadIdx.x;
    if (out_row < outH && out_col < outW) {
        scalar_t sum = 0;
        // The top-left corner of the pooling window in shared memory for this thread
        int sh_row_start = threadIdx.y * stride;
        int sh_col_start = threadIdx.x * stride;
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int sh_r = sh_row_start + i;
                int sh_c = sh_col_start + j;
                sum += shmem[sh_r * tileW + sh_c];
            }
        }
        int out_index = ((n * C + c) * outH + out_row) * outW + out_col;
        output[out_index] = sum / static_cast<scalar_t>(kernel_size * kernel_size);
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
    auto out = torch::empty({N, C, outH, outW}, x.options());

    // Define block size for spatial processing. Here we choose 16x16 threads per block.
    dim3 block(16, 16);
    // Grid dimensions tile the output spatially; each block processes a tile for one (n,c) plane.
    dim3 grid(
        (outH + block.y - 1) / block.y,
        (outW + block.x - 1) / block.x,
        N * C
    );

    // Compute shared memory size needed: tileH * tileW * sizeof(scalar_t).
    int tileH = block.y * stride + kernel_size - stride;
    int tileW = block.x * stride + kernel_size - stride;
    size_t shared_mem_size = tileH * tileW * sizeof(float);  // This will be overridden by sizeof(scalar_t) in the dispatch below.

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward_kernel", ([&] {
        avg_pool2d_forward_kernel<scalar_t><<<grid, block, tileH * tileW * sizeof(scalar_t)>>>(
            x_cont.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            N, C, H, W,
            outH, outW,
            kernel_size, stride, padding
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool2d_forward, "2D Average Pooling forward (CUDA) with Shared Memory");
}
