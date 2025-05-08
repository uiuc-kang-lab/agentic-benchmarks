#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

struct PoolConfig {
    int N, C, H, W, outH, outW;
    int kernel_size, stride, padding;
};

__constant__ PoolConfig cfg;

template <typename scalar_t>
__global__ void avg_pool2d_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output
) {
    extern __shared__ char shared_memory[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_memory);
    
    int nc = blockIdx.z;
    int n = nc / cfg.C;
    int c = nc % cfg.C;
    
    int h_out = blockIdx.x * blockDim.y + threadIdx.y;
    int w_out = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (n >= cfg.N || c >= cfg.C || h_out >= cfg.outH || w_out >= cfg.outW) return;

    // Calculate input region boundaries
    int h_start = h_out * cfg.stride - cfg.padding;
    int w_start = w_out * cfg.stride - cfg.padding;
    
    // Define shared memory tile size
    const int TILE_H = blockDim.y * cfg.stride + cfg.kernel_size - 1;
    const int TILE_W = blockDim.x * cfg.stride + cfg.kernel_size - 1;
    
    // Load input tile into shared memory
    for (int tile_h = threadIdx.y; tile_h < TILE_H; tile_h += blockDim.y) {
        for (int tile_w = threadIdx.x; tile_w < TILE_W; tile_w += blockDim.x) {
            int h_in = blockIdx.x * blockDim.y * cfg.stride - cfg.padding + tile_h;
            int w_in = blockIdx.y * blockDim.x * cfg.stride - cfg.padding + tile_w;
            
            if (h_in >= 0 && h_in < cfg.H && w_in >= 0 && w_in < cfg.W) {
                shared_input[tile_h * TILE_W + tile_w] = 
                    input[((n * cfg.C + c) * cfg.H + h_in) * cfg.W + w_in];
            } else {
                shared_input[tile_h * TILE_W + tile_w] = 0;
            }
        }
    }
    
    __syncthreads();

    // Compute average pooling using shared memory
    scalar_t sum_val = scalar_t(0);
    int tile_start_h = threadIdx.y * cfg.stride;
    int tile_start_w = threadIdx.x * cfg.stride;
    
    for (int i = 0; i < cfg.kernel_size; ++i) {
        for (int j = 0; j < cfg.kernel_size; ++j) {
            sum_val += shared_input[(tile_start_h + i) * TILE_W + (tile_start_w + j)];
        }
    }
    
    output[((n * cfg.C + c) * cfg.outH + h_out) * cfg.outW + w_out] = 
        sum_val / static_cast<scalar_t>(cfg.kernel_size * cfg.kernel_size);
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

    int outH = (H + 2 * padding - kernel_size)/stride + 1;
    int outW = (W + 2 * padding - kernel_size)/stride + 1;

    auto x_cont = x.contiguous();
    auto out = torch::empty({N, C, outH, outW}, x.options());

    PoolConfig host_cfg{N, C, H, W, outH, outW, kernel_size, stride, padding};
    cudaMemcpyToSymbol(cfg, &host_cfg, sizeof(PoolConfig));

    dim3 block(32, 4);
    dim3 grid(
        (outH + block.y - 1) / block.y,
        (outW + block.x - 1) / block.x,
        N * C
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool_forward", ([&] {
        avg_pool2d_forward_kernel<scalar_t><<<grid, block>>>(
            x_cont.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>()
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool2d_forward, "2D Average Pooling forward (CUDA)");
}