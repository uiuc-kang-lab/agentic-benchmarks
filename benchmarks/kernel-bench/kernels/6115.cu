#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool2d_forward_kernel_unroll(
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
    if (index >= total) {
        return;
    }

    int w_out = index % outW;
    int h_out = (index / outW) % outH;
    int c = (index / (outW * outH)) % C;
    int n = index / (outW * outH * C);

    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    scalar_t sum_val = scalar_t(0);
    #pragma unroll
    for (int i = 0; i < kernel_size; i++) {
        #pragma unroll
        for (int j = 0; j < kernel_size; j++) {
            int h_in = h_start + i;
            int w_in = w_start + j;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                sum_val += input[((n * C + c) * H + h_in) * W + w_in];
            }
        }
    }
    output[index] = sum_val / static_cast<scalar_t>(kernel_size * kernel_size);
}

torch::Tensor avg_pool2d_forward_unroll(
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

    const int threads = 256;
    const int blocks = (N * C * outH * outW + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward_kernel_unroll", ([&] {
        const scalar_t* input_data = x_cont.data_ptr<scalar_t>();
        scalar_t* output_data = out.data_ptr<scalar_t>();

        avg_pool2d_forward_kernel_unroll<<<blocks, threads>>>(
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
    m.def("forward", &avg_pool2d_forward_unroll, "2D Average Pooling forward with loop unrolling (CUDA)");
}