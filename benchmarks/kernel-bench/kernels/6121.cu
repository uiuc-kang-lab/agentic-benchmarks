#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses shared memory to load a tile of the input data that is reused
// for multiple output elements, reducing global memory latency.
// Each block computes a tile of output (8x8 by default) for a specific (n, c) slice.

template <typename scalar_t>
__global__ void avg_pool2d_forward_kernel_shared_tile(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int N,
    int C,
    int H,
    int W,
    int outH,
    int outW,
    int kernel_size,
    int stride,
    int padding
) {
    // Define tile dimensions for the output
    const int tile_out_w = 8;  // tile width
    const int tile_out_h = 8;  // tile height

    // Determine the starting output coordinates for this tile
    int out_tile_x = blockIdx.x * tile_out_w;
    int out_tile_y = blockIdx.y * tile_out_h;

    // blockIdx.z covers the combined (n, c) dimensions
    int linear_nc = blockIdx.z; // 0 <= linear_nc < N * C
    int n = linear_nc / C;
    int c = linear_nc % C;

    // Compute the dimensions of the shared memory tile.
    // The shared memory tile must cover all pooling windows for the output tile.
    int shared_tile_w = (tile_out_w - 1) * stride + kernel_size;
    int shared_tile_h = (tile_out_h - 1) * stride + kernel_size;

    // The top-left coordinate in the input corresponding to the shared tile
    int in_tile_x = out_tile_x * stride - padding;
    int in_tile_y = out_tile_y * stride - padding;

    // Allocate dynamic shared memory
    extern __shared__ char smem[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(smem);

    // Total number of elements in the shared tile
    int total_shared = shared_tile_w * shared_tile_h;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;

    // Each thread loads multiple elements from global memory to shared memory
    for (int index = tid; index < total_shared; index += num_threads) {
        int sm_y = index / shared_tile_w;
        int sm_x = index % shared_tile_w;
        int in_x = in_tile_x + sm_x;
        int in_y = in_tile_y + sm_y;
        if (in_x >= 0 && in_x < W && in_y >= 0 && in_y < H) {
            shared_input[index] = input[((n * C + c) * H + in_y) * W + in_x];
        } else {
            shared_input[index] = static_cast<scalar_t>(0);
        }
    }
    __syncthreads();

    // Each thread computes one output element for the tile
    int local_x = threadIdx.x;  // within [0, tile_out_w)
    int local_y = threadIdx.y;  // within [0, tile_out_h)
    int out_x = out_tile_x + local_x;
    int out_y = out_tile_y + local_y;

    if (out_x < outW && out_y < outH) {
        // Calculate the starting offset in the shared memory for the pooling window
        int sm_offset_x = local_x * stride;
        int sm_offset_y = local_y * stride;
        scalar_t sum_val = static_cast<scalar_t>(0);
        
        // Sum over the pooling window loaded in shared memory
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                int sm_idx = (sm_offset_y + i) * shared_tile_w + (sm_offset_x + j);
                sum_val += shared_input[sm_idx];
            }
        }
        
        // Write the average result to global memory
        output[((n * C + c) * outH + out_y) * outW + out_x] = sum_val / static_cast<scalar_t>(kernel_size * kernel_size);
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

    // Tile dimensions for the output computed per block
    const int tile_out_w = 8;
    const int tile_out_h = 8;
    dim3 block(tile_out_w, tile_out_h, 1);
    dim3 grid((outW + tile_out_w - 1) / tile_out_w,
              (outH + tile_out_h - 1) / tile_out_h,
              N * C);

    // Compute shared memory size needed per block
    int shared_tile_w = (tile_out_w - 1) * stride + kernel_size;
    int shared_tile_h = (tile_out_h - 1) * stride + kernel_size;
    size_t shared_mem_size = x.element_size() * shared_tile_w * shared_tile_h;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward_kernel_shared_tile", ([&] {
        avg_pool2d_forward_kernel_shared_tile<scalar_t><<<grid, block, shared_mem_size>>>(
            x_cont.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            N, C, H, W, outH, outW,
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
