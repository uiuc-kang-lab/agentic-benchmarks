#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel with focus on efficient thread and block indexing

template <typename scalar_t>
__global__ void avg_pool2d_forward_kernel_optimized(
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
    // Use 2D block and 2D grid configuration for better spatial locality
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;

    if (w_out >= outW || h_out >= outH) return;

    int c = blockIdx.z % C;
    int n = blockIdx.z / C;

    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    scalar_t sum_val = scalar_t(0);
    for (int i = 0; i < kernel_size; i++) {
        int h_in = h_start + i;
        if (h_in >= 0 && h_in < H) {
            for (int j = 0; j < kernel_size; j++) {
                int w_in = w_start + j;
                if (w_in >= 0 && w_in < W) {
                    sum_val += input[((n * C + c) * H + h_in) * W + w_in];
                }
            }
        }
    }
    output[((n * C + c) * outH + h_out) * outW + w_out] = sum_val / static_cast<scalar_t>(kernel_size * kernel_size);
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

    // Configure 2D grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y, N * C);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward_kernel_optimized", ([&] {
        const scalar_t* input_data = x_cont.data_ptr<scalar_t>();
        scalar_t* output_data = out.data_ptr<scalar_t>();

        avg_pool2d_forward_kernel_optimized<scalar_t><<<grid, block>>>(
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
    m.def("forward", &avg_pool2d_forward, "2D Average Pooling forward with optimized thread and block indexing (CUDA)");
}