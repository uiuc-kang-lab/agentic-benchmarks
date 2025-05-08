#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses grid-stride loops and manual unrolling (#pragma unroll) to reduce loop overhead.
// For the common case of 3x3 pooling when the pooling window is fully within the input bounds,
// the inner loops are completely unrolled, eliminating loop control overhead. For other cases,
// the kernel unrolls the inner loops to improve performance.

template <typename scalar_t>
__global__ void unrolled_avg_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int N,
    const int C,
    const int H,
    const int W,
    const int outH,
    const int outW,
    const int kernel_size,
    const int stride,
    const int padding
) {
    int total = N * C * outH * outW;
    int gridStride = blockDim.x * gridDim.x;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridStride) {
        // Compute output coordinates from flattened index
        int w_out = idx % outW;
        int h_out = (idx / outW) % outH;
        int c = (idx / (outW * outH)) % C;
        int n = idx / (outW * outH * C);

        int in_x_start = w_out * stride - padding;
        int in_y_start = h_out * stride - padding;
        scalar_t sum = static_cast<scalar_t>(0);

        // Fast path for 3x3 pooling if the window is fully inside the input
        if (kernel_size == 3 && in_x_start >= 0 && in_y_start >= 0 &&
            (in_x_start + 3) <= W && (in_y_start + 3) <= H) {
            int base = (n * C + c) * H;
            int row0 = base + in_y_start;
            int row1 = row0 + 1;
            int row2 = row0 + 2;
            int ix = in_x_start;
            sum = input[row0 * W + ix]     + input[row0 * W + ix + 1]     + input[row0 * W + ix + 2] +
                  input[row1 * W + ix]     + input[row1 * W + ix + 1]     + input[row1 * W + ix + 2] +
                  input[row2 * W + ix]     + input[row2 * W + ix + 1]     + input[row2 * W + ix + 2];
        } else {
            // Generic path with loop unrolling for inner loops
            #pragma unroll
            for (int ky = 0; ky < kernel_size; ky++) {
                int y = in_y_start + ky;
                if (y >= 0 && y < H) {
                    int offset = ((n * C + c) * H + y) * W;
                    #pragma unroll
                    for (int kx = 0; kx < kernel_size; kx++) {
                        int x = in_x_start + kx;
                        if (x >= 0 && x < W) {
                            sum += input[offset + x];
                        }
                    }
                }
            }
        }
        output[idx] = sum / static_cast<scalar_t>(kernel_size * kernel_size);
    }
}

// Forward function exposed to PyTorch

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

    auto x_contiguous = x.contiguous();
    auto output = torch::empty({N, C, outH, outW}, x.options());

    int total = N * C * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "unrolled_avg_pool2d_kernel", ([&] {
        unrolled_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            x_contiguous.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C, H, W,
            outH, outW,
            kernel_size, stride, padding
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &unrolled_avg_pool2d_forward, "Unrolled 2D Average Pooling forward (CUDA)");
}
