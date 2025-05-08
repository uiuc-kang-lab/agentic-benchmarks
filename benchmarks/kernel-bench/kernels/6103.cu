#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define constant memory for frequently accessed, read-only parameters
__constant__ int d_N;
__constant__ int d_C;
__constant__ int d_H;
__constant__ int d_W;
__constant__ int d_outH;
__constant__ int d_outW;
__constant__ int d_kernel_size;
__constant__ int d_stride;
__constant__ int d_padding;
__constant__ float d_inv_kernel_area;

// Kernel using grid-stride loop and reading parameters from constant memory
template <typename scalar_t>
__global__ void constant_memory_avg_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;
    for (; idx < total; idx += gridStride) {
        // Compute output coordinates
        int w_out = idx % d_outW;
        int h_out = (idx / d_outW) % d_outH;
        int nc = idx / (d_outW * d_outH);
        int c = nc % d_C;
        int n = nc / d_C;

        // Compute input starting indices
        int in_x_start = w_out * d_stride - d_padding;
        int in_y_start = h_out * d_stride - d_padding;
        scalar_t sum = static_cast<scalar_t>(0);

        // Fast path: 3x3 pooling when window is fully inside input
        if (d_kernel_size == 3 && in_x_start >= 0 && in_y_start >= 0 &&
            (in_x_start + 3) <= d_W && (in_y_start + 3) <= d_H) {
            int base = (n * d_C + c) * d_H;
            int row0 = base + in_y_start;
            int row1 = row0 + 1;
            int row2 = row0 + 2;
            sum = input[row0 * d_W + in_x_start]     + input[row0 * d_W + in_x_start + 1]     + input[row0 * d_W + in_x_start + 2] +
                  input[row1 * d_W + in_x_start]     + input[row1 * d_W + in_x_start + 1]     + input[row1 * d_W + in_x_start + 2] +
                  input[row2 * d_W + in_x_start]     + input[row2 * d_W + in_x_start + 1]     + input[row2 * d_W + in_x_start + 2];
        } else {
            // Generic path with boundary checks
            for (int ky = 0; ky < d_kernel_size; ky++) {
                int y = in_y_start + ky;
                if (y >= 0 && y < d_H) {
                    int offset = ((n * d_C + c) * d_H + y) * d_W;
                    for (int kx = 0; kx < d_kernel_size; kx++) {
                        int x = in_x_start + kx;
                        if (x >= 0 && x < d_W) {
                            sum += input[offset + x];
                        }
                    }
                }
            }
        }

        output[idx] = sum * d_inv_kernel_area;
    }
}

// Forward function exposed to PyTorch
torch::Tensor constant_memory_avg_pool2d_forward(
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
    auto output = torch::empty({N, C, outH, outW}, x.options());

    // Copy frequently accessed parameters to constant memory
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));
    cudaMemcpyToSymbol(d_C, &C, sizeof(int));
    cudaMemcpyToSymbol(d_H, &H, sizeof(int));
    cudaMemcpyToSymbol(d_W, &W, sizeof(int));
    cudaMemcpyToSymbol(d_outH, &outH, sizeof(int));
    cudaMemcpyToSymbol(d_outW, &outW, sizeof(int));
    cudaMemcpyToSymbol(d_kernel_size, &kernel_size, sizeof(int));
    cudaMemcpyToSymbol(d_stride, &stride, sizeof(int));
    cudaMemcpyToSymbol(d_padding, &padding, sizeof(int));
    float inv_kernel_area = 1.0f / (kernel_size * kernel_size);
    cudaMemcpyToSymbol(d_inv_kernel_area, &inv_kernel_area, sizeof(float));

    int total = N * C * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "constant_memory_avg_pool2d_kernel", ([&] {
        constant_memory_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            x_cont.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &constant_memory_avg_pool2d_forward, "Constant Memory 2D Average Pooling forward (CUDA)");
}
