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
    // Align thread indices to promote coalesced memory access
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride_x = blockDim.x * gridDim.x;
    const int total = N * C * outH * outW;

    for (int index = tid; index < total; index += stride_x) {
        const int w_out = index % outW;
        const int h_out = (index / outW) % outH;
        const int c = (index / (outW * outH)) % C;
        const int n = index / (outW * outH * C);

        const int h_start = h_out * stride - padding;
        const int w_start = w_out * stride - padding;

        scalar_t sum_val = scalar_t(0);
        const scalar_t scale = scalar_t(1.0) / (kernel_size * kernel_size);

        #pragma unroll
        for (int i = 0; i < kernel_size; i++) {
            const int h_in = h_start + i;
            if (h_in >= 0 && h_in < H) {
                #pragma unroll
                for (int j = 0; j < kernel_size; j++) {
                    const int w_in = w_start + j;
                    if (w_in >= 0 && w_in < W) {
                        // Use __ldg for read-only memory access
                        sum_val += __ldg(&input[((n * C + c) * H + h_in) * W + w_in]);
                    }
                }
            }
        }
        output[index] = sum_val * scale;
    }
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

    // Optimize block and grid dimensions for better occupancy
    const int threads = 256;
    const int blocks = std::min(65535, int((N * C * outH * outW + threads - 1) / threads));

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