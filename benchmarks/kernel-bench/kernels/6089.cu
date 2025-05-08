#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel manually unrolls the inner loops for the common 3x3 pooling case
// to reduce loop overhead. For other kernel sizes or boundary cases, it falls back
// to a generic loop with #pragma unroll hints.

template <typename scalar_t>
__global__ void manual_unroll_avg_pool2d_kernel(
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
    // Map threads to output pixel positions using a 2D grid and use blockIdx.z for combined (n, c)
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc % C;

    if (out_x >= outW || out_y >= outH)
        return;

    // Compute starting point in the input tensor
    int in_x_start = out_x * stride - padding;
    int in_y_start = out_y * stride - padding;
    scalar_t sum = scalar_t(0);

    // Check if the pooling window is completely inside the input
    bool fully_inside = (in_x_start >= 0) && (in_y_start >= 0) &&
                        ((in_x_start + kernel_size) <= W) &&
                        ((in_y_start + kernel_size) <= H);

    // Compute the output index
    int out_index = ((n * C + c) * outH + out_y) * outW + out_x;

    // Fast path: if kernel_size is 3 and the window is fully inside, manually unroll loops
    if (kernel_size == 3 && fully_inside) {
        int base = (n * C + c) * H;
        int ix = in_x_start;
        int row0 = base + in_y_start;
        int row1 = base + in_y_start + 1;
        int row2 = base + in_y_start + 2;
        sum = input[row0 * W + ix]     + input[row0 * W + ix + 1]     + input[row0 * W + ix + 2] +
              input[row1 * W + ix]     + input[row1 * W + ix + 1]     + input[row1 * W + ix + 2] +
              input[row2 * W + ix]     + input[row2 * W + ix + 1]     + input[row2 * W + ix + 2];
    } else {
        // Generic path with #pragma unroll hint for small kernel sizes
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ky++) {
            int y = in_y_start + ky;
            #pragma unroll
            for (int kx = 0; kx < kernel_size; kx++) {
                int x = in_x_start + kx;
                if (y >= 0 && y < H && x >= 0 && x < W) {
                    int index_in = ((n * C + c) * H + y) * W + x;
                    sum += input[index_in];
                }
            }
        }
    }

    output[out_index] = sum / static_cast<scalar_t>(kernel_size * kernel_size);
}

// Forward function exposed to PyTorch

torch::Tensor manual_unroll_avg_pool2d_forward(
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
    auto options = x.options();
    auto output = torch::empty({N, C, outH, outW}, options);
    
    // Use a 2D block for spatial dimensions and gridDim.z for the combined N*C dimension
    dim3 threads(32, 8);
    dim3 blocks(
        (outW + threads.x - 1) / threads.x,
        (outH + threads.y - 1) / threads.y,
        N * C
    );
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "manual_unroll_avg_pool2d_kernel", ([&] {
        manual_unroll_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &manual_unroll_avg_pool2d_forward, "Manual Unroll 2D Average Pooling forward (CUDA)");
}
