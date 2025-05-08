#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel implements 2D average pooling with branchless boundary checks
// to minimize warp divergence. All threads follow a uniform control flow using
// a grid-stride loop and a branchless ternary operator in the inner loop.

template <typename scalar_t>
__global__ void branchless_avg_pool2d_kernel(
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;

    for (; idx < total; idx += gridStride) {
        // Compute output indices from the flattened index
        int w_out = idx % outW;
        int h_out = (idx / outW) % outH;
        int c = (idx / (outW * outH)) % C;
        int n = idx / (outW * outH * C);

        // Compute starting input indices
        int in_x_start = w_out * stride - padding;
        int in_y_start = h_out * stride - padding;
        scalar_t sum = static_cast<scalar_t>(0);

        // Process kernel window uniformly using branchless conditional
        for (int ky = 0; ky < kernel_size; ky++) {
            int y = in_y_start + ky;
            for (int kx = 0; kx < kernel_size; kx++) {
                int x = in_x_start + kx;
                // Use ternary operator to avoid divergent branches.
                // The conditional operator ensures that out-of-bound accesses are not performed.
                sum += ((y >= 0 && y < H && x >= 0 && x < W) ? 
                        input[((n * C + c) * H + y) * W + x] : static_cast<scalar_t>(0));
            }
        }

        // Write the result
        output[idx] = sum / static_cast<scalar_t>(kernel_size * kernel_size);
    }
}

// Forward function exposed to PyTorch

torch::Tensor branchless_avg_pool2d_forward(
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

    auto x_contiguous = x.contiguous();
    auto output = torch::empty({N, C, outH, outW}, x.options());

    int total = N * C * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "branchless_avg_pool2d_kernel", ([&] {
        branchless_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &branchless_avg_pool2d_forward, "Branchless 2D Average Pooling forward (CUDA)");
}
