#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int KERNEL_SIZE=3>
__global__ void optimized_avg_pool2d_kernel(
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
    __shared__ scalar_t shared_input[32][32];
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    
    const scalar_t inv_window_size = scalar_t(1.0) / (KERNEL_SIZE * KERNEL_SIZE);

    for (int idx = index; idx < total; idx += blockDim.x * gridDim.x) {
        const int w_out = idx % outW;
        const int h_out = (idx / outW) % outH;
        const int c = (idx / (outW * outH)) % C;
        const int n = idx / (outW * outH * C);

        const int h_base = h_out * stride - padding;
        const int w_base = w_out * stride - padding;

        const int h_start = max(0, h_base);
        const int w_start = max(0, w_base);
        const int h_end = min(H, h_base + KERNEL_SIZE);
        const int w_end = min(W, w_base + KERNEL_SIZE);

        scalar_t sum_val = scalar_t(0);
        
        #pragma unroll
        for (int h = h_start; h < h_end; h++) {
            const int base_offset = ((n * C + c) * H + h) * W;
            
            #pragma unroll
            for (int w = w_start; w < w_end; w++) {
                sum_val += input[base_offset + w];
            }
        }
        
        output[idx] = sum_val * inv_window_size;
    }
}

torch::Tensor optimized_avg_pool2d_forward(
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
    const int outH = (H + 2 * padding - kernel_size) / stride + 1;
    const int outW = (W + 2 * padding - kernel_size) / stride + 1;

    auto x_cont = x.contiguous();
    auto options = x.options();
    auto out = torch::empty({N, C, outH, outW}, options);

    const int threads = 256;
    const int blocks = min(65535, (N * C * outH * outW + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "optimized_avg_pool2d_kernel", ([&] {
        optimized_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_avg_pool2d_forward, "Optimized 2D Average Pooling forward (CUDA)");
}