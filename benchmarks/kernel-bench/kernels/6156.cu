#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tiled shared memory implementation for avg_pool2d_forward

#define TILE_W 16
#define TILE_H 16

template <typename scalar_t>
__global__ void avg_pool2d_forward_kernel_tiled(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int H,
    int W,
    int outH,
    int outW,
    int kernel_size,
    int stride,
    int padding,
    int C
) {
    // Compute n and c from blockIdx.z; each block in z dimension corresponds to one (n, c) pair
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc %% C;  // use %% to escape % in replacement

    // Determine the starting output coordinates for this tile
    int out_tile_x = blockIdx.x * TILE_W;
    int out_tile_y = blockIdx.y * TILE_H;

    // Thread indices within the tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Dimensions of the shared memory tile
    int shared_w = TILE_W * stride + (kernel_size - stride);
    int shared_h = TILE_H * stride + (kernel_size - stride);

    // Declare shared memory dynamically
    extern __shared__ char smem_char[];
    scalar_t* shmem = reinterpret_cast<scalar_t*>(smem_char);

    // Load the required patch from global memory into shared memory
    // Each thread loads multiple elements if necessary
    for (int y = ty; y < shared_h; y += blockDim.y) {
        for (int x = tx; x < shared_w; x += blockDim.x) {
            int in_y = out_tile_y * stride - padding + y;
            int in_x = out_tile_x * stride - padding + x;
            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                shmem[y * shared_w + x] = input[((n * C + c) * H + in_y) * W + in_x];
            } else {
                shmem[y * shared_w + x] = scalar_t(0);
            }
        }
    }
    __syncthreads();

    // Compute the output if within bounds
    int out_x = out_tile_x + tx;
    int out_y = out_tile_y + ty;
    if (tx < TILE_W && ty < TILE_H && out_x < outW && out_y < outH) {
        int shmem_x = tx * stride;
        int shmem_y = ty * stride;
        scalar_t sum_val = 0;
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                sum_val += shmem[(shmem_y + i) * shared_w + (shmem_x + j)];
            }
        }
        int index = ((n * C) * outH + out_y) * outW + out_x;
        output[index] = sum_val / static_cast<scalar_t>(kernel_size * kernel_size);
    }
}

torch::Tensor avg_pool2d_forward(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding
) {
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor.");
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    int outH = (H + 2 * padding - kernel_size) / stride + 1;
    int outW = (W + 2 * padding - kernel_size) / stride + 1;

    auto x_cont = x.contiguous();
    auto options = x.options();
    auto out = torch::empty({N, C, outH, outW}, options);

    const int threads = 128;
    const int blocks = (N * C * outH * outW + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward_kernel", ([&] {
        const scalar_t* input_data = x_cont.data_ptr<scalar_t>();
        scalar_t* output_data = out.data_ptr<scalar_t>();

        avg_pool2d_forward_kernel<<<blocks, threads>>>(
            input_data,
            output_data,
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
    m.def("forward", &avg_pool2d_forward, "2D Average Pooling forward (CUDA)");
}