#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 16

template <typename scalar_t>
__global__ void avg_pool2d_forward_shared_kernel(
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
    __shared__ scalar_t tile[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int w_out = blockIdx.x * blockDim.x + tx;
    int h_out = blockIdx.y * blockDim.y + ty;
    int c = blockIdx.z;

    if (w_out < outW && h_out < outH) {
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;

        scalar_t sum_val = scalar_t(0);

        for (int i = 0; i < kernel_size; i += TILE_DIM) {
            for (int j = 0; j < kernel_size; j += TILE_DIM) {
                int h_in = h_start + i + ty;
                int w_in = w_start + j + tx;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    tile[ty][tx] = input[((blockIdx.z * H + h_in) * W + w_in)];
                } else {
                    tile[ty][tx] = 0;
                }
                __syncthreads();

                if (ty == 0 && tx == 0) {
                    for (int m = 0; m < TILE_DIM; ++m) {
                        for (int n = 0; n < TILE_DIM; ++n) {
                            sum_val += tile[m][n];
                        }
                    }
                }
                __syncthreads();
            }
        }

        if (ty == 0 && tx == 0) {
            output[(blockIdx.z * outH * outW) + (h_out * outW) + w_out] = sum_val / static_cast<scalar_t>(kernel_size * kernel_size);
        }
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

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((outW + TILE_DIM - 1) / TILE_DIM, (outH + TILE_DIM - 1) / TILE_DIM, C);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward_shared_kernel", ([&] {
        const scalar_t* input_data = x_cont.data_ptr<scalar_t>();
        scalar_t* output_data = out.data_ptr<scalar_t>();

        avg_pool2d_forward_shared_kernel<<<blocks, threads>>>(
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
    m.def("forward", &avg_pool2d_forward, "2D Average Pooling forward using shared memory (CUDA)");
}