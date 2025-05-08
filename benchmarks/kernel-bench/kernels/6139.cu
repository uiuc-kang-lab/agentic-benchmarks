#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int KERNEL_SIZE=3>
__global__ void avg_pool2d_forward_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int N,
    const int C,
    const int H,
    const int W,
    const int outH,
    const int outW,
    const int stride,
    const int padding
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * outH * outW;
    if (index >= total) return;

    const int w_out = index % outW;
    const int h_out = (index / outW) % outH;
    const int c = (index / (outW * outH)) % C;
    const int n = index / (outW * outH * C);

    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;
    
    const scalar_t scale = scalar_t(1.0) / (KERNEL_SIZE * KERNEL_SIZE);
    
    scalar_t sum_val = 0;
    
    #pragma unroll
    for (int i = 0; i < KERNEL_SIZE; i++) {
        const int h_in = h_start + i;
        #pragma unroll
        for (int j = 0; j < KERNEL_SIZE; j++) {
            const int w_in = w_start + j;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                sum_val += input[((n * C + c) * H + h_in) * W + w_in];
            }
        }
    }
    
    output[index] = sum_val * scale;
}

torch::Tensor avg_pool2d_forward_optimized(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding
) {
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor.");
    TORCH_CHECK(kernel_size == 3, "This optimized version only supports 3x3 kernel size");
    
    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);

    const int outH = (H + 2 * padding - kernel_size) / stride + 1;
    const int outW = (W + 2 * padding - kernel_size) / stride + 1;

    auto x_cont = x.contiguous();
    auto options = x.options();
    auto out = torch::empty({N, C, outH, outW}, options);

    const int threads = 256;
    const int blocks = (N * C * outH * outW + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward_kernel_optimized", ([&] {
        avg_pool2d_forward_kernel_optimized<scalar_t, 3><<<blocks, threads>>>(
            x_cont.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            N, C, H, W,
            outH, outW,
            stride, padding
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return out;
}