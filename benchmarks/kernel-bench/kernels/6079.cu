#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses loop unrolling to reduce loop overhead and improve performance.

template <typename scalar_t>
__global__ void unrolled_avg_pool2d_kernel(
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
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    int thread_stride = blockDim.x * gridDim.x;

    for (int idx = index; idx < total; idx += thread_stride) {
        int w_out = idx % outW;
        int h_out = (idx / outW) % outH;
        int c = (idx / (outW * outH)) % C;
        int n = idx / (outW * outH * C);

        // Compute base coordinates in the input
        int h_base = h_out * stride - padding;
        int w_base = w_out * stride - padding;

        // Precompute effective boundaries to avoid per-iteration conditionals
        int h_start = (h_base < 0) ? 0 : h_base;
        int w_start = (w_base < 0) ? 0 : w_base;
        int h_end = (h_base + kernel_size > H) ? H : (h_base + kernel_size);
        int w_end = (w_base + kernel_size > W) ? W : (w_base + kernel_size);

        scalar_t sum_val = scalar_t(0);

        // Unroll the loops to improve performance
        #pragma unroll
        for (int h = h_start; h < h_end; h++) {
            int base_offset = ((n * C + c) * H + h) * W;
            #pragma unroll
            for (int w = w_start; w < w_end; w++) {
                sum_val += input[base_offset + w];
            }
        }
        
        // Divide by the constant pooling window area
        output[idx] = sum_val / static_cast<scalar_t>(kernel_size * kernel_size);
    }
}

torch::Tensor unrolled_avg_pool2d_forward(
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

    const int threads = 256;
    const int blocks = (N * C * outH * outW + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "unrolled_avg_pool2d_kernel", ([&] {
        const scalar_t* input_data = x_cont.data_ptr<scalar_t>();
        scalar_t* output_data = out.data_ptr<scalar_t>();

        unrolled_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &unrolled_avg_pool2d_forward, "Unrolled 2D Average Pooling forward (CUDA)");
}
