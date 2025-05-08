#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int KERNEL_SIZE=0>
__global__ void grid_stride_avg_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int N,
    const int C,
    const int H,
    const int W,
    const int outH,
    const int outW,
    const int stride,
    const int padding,
    const int actual_kernel_size
) {
    const int total_elements = N * C * outH * outW;
    const int grid_stride = blockDim.x * gridDim.x;
    const float inv_window_size = 1.0f / (actual_kernel_size * actual_kernel_size);

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_elements; 
         idx += grid_stride) {
        
        const int w_out = idx % outW;
        const int h_out = (idx / outW) % outH;
        const int c = (idx / (outW * outH)) % C;
        const int n = idx / (outW * outH * C);

        const int h_start = h_out * stride - padding;
        const int w_start = w_out * stride - padding;
        
        const int h_end = h_start + actual_kernel_size;
        const int w_end = w_start + actual_kernel_size;

        float sum = 0.0f;
        const int h_clamp_start = max(0, h_start);
        const int w_clamp_start = max(0, w_start);
        const int h_clamp_end = min(H, h_end);
        const int w_clamp_end = min(W, w_end);

        if (KERNEL_SIZE > 0) {
            #pragma unroll
            for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
                const int h = h_start + kh;
                if (h >= 0 && h < H) {
                    const int row_offset = ((n * C + c) * H + h) * W;
                    #pragma unroll
                    for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                        const int w = w_start + kw;
                        if (w >= 0 && w < W) {
                            sum += input[row_offset + w];
                        }
                    }
                }
            }
        } else {
            for (int h = h_clamp_start; h < h_clamp_end; ++h) {
                const int row_offset = ((n * C + c) * H + h) * W;
                for (int w = w_clamp_start; w < w_clamp_end; ++w) {
                    sum += input[row_offset + w];
                }
            }
        }

        output[idx] = sum * inv_window_size;
    }
}

torch::Tensor grid_stride_avg_pool2d_forward(
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
    auto out = torch::empty({N, C, outH, outW}, x.options());

    const int total = N * C * outH * outW;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "grid_stride_avg_pool2d", [&] {
        if (kernel_size == 3) {
            grid_stride_avg_pool2d_kernel<scalar_t, 3><<<blocks, threads>>>(
                x_cont.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                N, C, H, W,
                outH, outW,
                stride, padding,
                kernel_size
            );
        } else {
            grid_stride_avg_pool2d_kernel<scalar_t, 0><<<blocks, threads>>>(
                x_cont.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                N, C, H, W,
                outH, outW,
                stride, padding,
                kernel_size
            );
        }
    });

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &grid_stride_avg_pool2d_forward, "Grid-Stride 2D Average Pooling forward (CUDA)");
}