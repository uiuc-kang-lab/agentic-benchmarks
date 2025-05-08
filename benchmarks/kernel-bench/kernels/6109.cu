#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses a flat 1D grid-stride loop to evenly distribute workload among threads

template <typename scalar_t>
__global__ void even_workload_avg_pool2d_kernel(
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
    
    // Each thread handles multiple output elements via grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridStride) {
        // Compute the output coordinates
        int w_out = idx % outW;
        int h_out = (idx / outW) % outH;
        int c = (idx / (outW * outH)) % C;
        int n = idx / (outW * outH * C);
        
        // Calculate the corresponding top-left corner of the pooling window in input
        int in_x = w_out * stride - padding;
        int in_y = h_out * stride - padding;

        scalar_t sum = static_cast<scalar_t>(0);

        // Fast path for common 3x3 pooling when the entire window is within bounds
        if (kernel_size == 3 && in_x >= 0 && in_y >= 0 && (in_x + 3) <= W && (in_y + 3) <= H) {
            int base = (n * C + c) * H;
            int row0 = base + in_y;
            int row1 = row0 + 1;
            int row2 = row0 + 2;
            sum = input[row0 * W + in_x]     + input[row0 * W + in_x + 1]     + input[row0 * W + in_x + 2] +
                  input[row1 * W + in_x]     + input[row1 * W + in_x + 1]     + input[row1 * W + in_x + 2] +
                  input[row2 * W + in_x]     + input[row2 * W + in_x + 1]     + input[row2 * W + in_x + 2];
        } else {
            // Generic path: iterate over the pooling window with boundary checks
            for (int ky = 0; ky < kernel_size; ++ky) {
                int y = in_y + ky;
                if (y < 0 || y >= H) continue;
                int row_offset = ((n * C + c) * H + y) * W;
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int x = in_x + kx;
                    if (x < 0 || x >= W) continue;
                    sum += input[row_offset + x];
                }
            }
        }

        output[idx] = sum / static_cast<scalar_t>(kernel_size * kernel_size);
    }
}

// Forward function exposed to PyTorch

torch::Tensor even_workload_avg_pool2d_forward(
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
    
    // Compute output dimensions
    int outH = (H + 2 * padding - kernel_size) / stride + 1;
    int outW = (W + 2 * padding - kernel_size) / stride + 1;
    
    auto x_cont = x.contiguous();
    auto output = torch::empty({N, C, outH, outW}, x.options());
    
    int total = N * C * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "even_workload_avg_pool2d_kernel", ([&] {
        even_workload_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            x_cont.data_ptr<scalar_t>(),
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
    m.def("forward", &even_workload_avg_pool2d_forward, "Evenly Distributed Average Pooling forward (CUDA)");
}
