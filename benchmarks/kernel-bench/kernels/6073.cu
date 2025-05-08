#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool2d_forward_kernel(
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
    // Calculate position based on width-first ordering for coalesced access
    const int w_out = threadIdx.x + blockIdx.x * blockDim.x;
    const int h_out = blockIdx.y % outH;
    const int c = (blockIdx.y / outH) % C;
    const int n = blockIdx.y / (outH * C);

    if (w_out >= outW || n >= N) return;

    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;
    
    scalar_t sum_val = 0;
    #pragma unroll
    for (int i = 0; i < kernel_size; i++) {
        const int h_in = h_start + i;
        if (h_in >= 0 && h_in < H) {
            #pragma unroll
            for (int j = 0; j < kernel_size; j++) {
                const int w_in = w_start + j;
                if (w_in >= 0 && w_in < W) {
                    sum_val += input[((n * C + c) * H + h_in) * W + w_in];
                }
            }
        }
    }

    const int out_idx = ((n * C + c) * outH + h_out) * outW + w_out;
    output[out_idx] = sum_val / static_cast<scalar_t>(kernel_size * kernel_size);
}

torch::Tensor avg_pool2d_forward(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding
) {
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor.");
    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);

    const int outH = (H + 2 * padding - kernel_size) / stride + 1;
    const int outW = (W + 2 * padding - kernel_size) / stride + 1;

    auto x_cont = x.contiguous();
    auto options = x.options();
    auto out = torch::empty({N, C, outH, outW}, options);

    const int threads_x = 256;
    const int blocks_x = (outW + threads_x - 1) / threads_x;
    const int blocks_y = N * C * outH;

    dim3 threads(threads_x);
    dim3 blocks(blocks_x, blocks_y);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward_kernel", ([&] {
        avg_pool2d_forward_kernel<<<blocks, threads>>>(
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
    m.def("forward", &avg_pool2d_forward, "2D Average Pooling forward (CUDA)");
}