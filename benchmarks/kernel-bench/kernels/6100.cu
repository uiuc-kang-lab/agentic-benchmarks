/*
Combined CUDA kernel for 2D average pooling using grid‐stride loops and manual unrolling for the common 3x3 case.
This implementation combines the strengths of two prior kernels:
1. The grid‐stride loop from Kernel 2 enables a lightweight 1D launch configuration over all output elements.
2. The manual unrolling for the 3x3 case (from Kernel 1) is used when the pooling window lies fully inside the input, reducing loop overhead.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__global__ void combined_avg_pool2d_kernel(
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
    // Total number of output elements
    int total = N * C * outH * outW;
    int gridStride = blockDim.x * gridDim.x;

    // Grid-stride loop over flattened output elements
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridStride) {
        // Compute output coordinates from the flattened index
        int w_out = idx % outW;
        int h_out = (idx / outW) % outH;
        int c = (idx / (outW * outH)) % C;
        int n = idx / (outW * outH * C);

        // Compute the corresponding starting position in the input
        int in_x_start = w_out * stride - padding;
        int in_y_start = h_out * stride - padding;
        scalar_t sum = scalar_t(0);

        // Fast path for 3x3 pooling when the window is fully inside the input bounds
        if (kernel_size == 3 && in_x_start >= 0 && in_y_start >= 0 &&
            (in_x_start + 3) <= W && (in_y_start + 3) <= H) {
            // Compute the base row index for this (n, c) slice
            int base = (n * C + c) * H;
            int row0 = base + in_y_start;
            int row1 = row0 + 1;
            int row2 = row0 + 2;
            // Manual unrolling over the 3x3 window
            sum = input[row0 * W + in_x_start]     + input[row0 * W + in_x_start + 1]     + input[row0 * W + in_x_start + 2] +
                  input[row1 * W + in_x_start]     + input[row1 * W + in_x_start + 1]     + input[row1 * W + in_x_start + 2] +
                  input[row2 * W + in_x_start]     + input[row2 * W + in_x_start + 1]     + input[row2 * W + in_x_start + 2];
        } else {
            // Generic path with nested loops and boundary checks
            for (int ky = 0; ky < kernel_size; ky++) {
                int y = in_y_start + ky;
                if (y >= 0 && y < H) {
                    int offset = ((n * C + c) * H + y) * W;
                    for (int kx = 0; kx < kernel_size; kx++) {
                        int x = in_x_start + kx;
                        if (x >= 0 && x < W) {
                            sum += input[offset + x];
                        }
                    }
                }
            }
        }
        
        // Write the averaged result to the output (division by window area)
        output[idx] = sum / static_cast<scalar_t>(kernel_size * kernel_size);
    }
}


// Forward function exposed to PyTorch

torch::Tensor combined_avg_pool2d_forward(
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

    // Compute output dimensions
    const int outH = (H + 2 * padding - kernel_size) / stride + 1;
    const int outW = (W + 2 * padding - kernel_size) / stride + 1;

    auto x_cont = x.contiguous();
    auto output = torch::empty({N, C, outH, outW}, x.options());

    // Total number of output elements
    int total = N * C * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "combined_avg_pool2d_kernel", ([&] {
        combined_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            x_cont.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C, H, W,
            outH, outW,
            kernel_size,
            stride,
            padding
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));
    
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &combined_avg_pool2d_forward, "Combined Grid-Stride and Manual Unroll 2D Average Pooling forward (CUDA)");
}
