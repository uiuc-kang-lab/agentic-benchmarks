#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized 2D Average Pooling Kernel with 2D thread indexing

template <typename scalar_t>
__global__ void avg_pool2d_forward_kernel_fast(
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
    // Use 2D thread indexing for spatial dims; each block in z handles a specific (n, c)
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;  // combined index for batch and channel

    if (nc >= N * C) return;
    int n = nc / C;
    int c = nc % C;

    if (h_out < outH && w_out < outW) {
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;

        scalar_t sum_val = scalar_t(0);
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                int h_in = h_start + i;
                int w_in = w_start + j;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int input_index = ((n * C + c) * H + h_in) * W + w_in;
                    sum_val += input[input_index];
                }
            }
        }
        int output_index = ((n * C + c) * outH + h_out) * outW + w_out;
        output[output_index] = sum_val / static_cast<scalar_t>(kernel_size * kernel_size);
    }
}


torch::Tensor avg_pool2d_forward(
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

    // Use a 2D configuration for spatial dimensions and map (n, c) to the third grid dimension
    dim3 threads(16, 16);
    dim3 blocks((outW + threads.x - 1) / threads.x,
                (outH + threads.y - 1) / threads.y,
                N * C);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward_kernel_fast", ([&] {
        const scalar_t* input_data = x_cont.data_ptr<scalar_t>();
        scalar_t* output_data = out.data_ptr<scalar_t>();
        avg_pool2d_forward_kernel_fast<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &avg_pool2d_forward, "Fast 2D Average Pooling forward (CUDA)");
}
