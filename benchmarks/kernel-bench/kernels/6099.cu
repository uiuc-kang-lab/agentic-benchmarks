#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized avg pool2d kernel that combines fast and generic paths
// using manual unrolling for the common 3x3 case, a generic fast-path when the pooling
// window is fully inside the input, and boundary checks otherwise.

template <typename scalar_t>
__global__ void efficient_avg_pool2d_kernel(
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
    // Map threads to the spatial output dimensions
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int out_x = blockIdx.x * blockDim.x + tid_x;
    const int out_y = blockIdx.y * blockDim.y + tid_y;

    // blockIdx.z covers the combined batch and channel dimensions
    const int nc = blockIdx.z;
    const int n = nc / C;
    const int c = nc % C;

    if (out_x >= outW || out_y >= outH) {
        return;
    }

    // Calculate starting position of the pooling window
    const int in_x_start = out_x * stride - padding;
    const int in_y_start = out_y * stride - padding;
    scalar_t sum = static_cast<scalar_t>(0);

    // Check if the entire pooling window is within input boundaries
    bool fully_inside = (in_x_start >= 0) && (in_y_start >= 0) &&
                        ((in_x_start + kernel_size) <= W) &&
                        ((in_y_start + kernel_size) <= H);

    // Compute the output index
    const int out_index = ((n * C + c) * outH + out_y) * outW + out_x;

    // Fast-path for the common 3x3 pooling case using manual unrolling
    if (kernel_size == 3 && fully_inside) {
        int base = (n * C + c) * H;
        int row0 = base + in_y_start;
        int row1 = base + in_y_start + 1;
        int row2 = base + in_y_start + 2;
        int ix = in_x_start;
        sum = input[row0 * W + ix]     + input[row0 * W + ix + 1]     + input[row0 * W + ix + 2] +
              input[row1 * W + ix]     + input[row1 * W + ix + 1]     + input[row1 * W + ix + 2] +
              input[row2 * W + ix]     + input[row2 * W + ix + 1]     + input[row2 * W + ix + 2];
    } else if (fully_inside) {
        // Generic fast-path: pooling window is fully inside, use unrolled loops
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ky++) {
            int row = in_y_start + ky;
            int offset = ((n * C + c) * H + row) * W + in_x_start;
            #pragma unroll
            for (int kx = 0; kx < kernel_size; kx++) {
                sum += input[offset + kx];
            }
        }
    } else {
        // Boundary path: check each element if it's within the input bounds
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ky++) {
            int y = in_y_start + ky;
            #pragma unroll
            for (int kx = 0; kx < kernel_size; kx++) {
                int x = in_x_start + kx;
                if (y >= 0 && y < H && x >= 0 && x < W) {
                    int idx = ((n * C + c) * H + y) * W + x;
                    sum += input[idx];
                }
            }
        }
    }

    output[out_index] = sum / static_cast<scalar_t>(kernel_size * kernel_size);
}

// Forward function exposed to PyTorch

torch::Tensor efficient_avg_pool2d_forward(
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
    auto options = x.options();
    auto output = torch::empty({N, C, outH, outW}, options);

    // Configure threads and blocks: use a 2D grid for spatial dimensions and blockIdx.z for the combined N*C dimension
    dim3 threads(32, 8);
    dim3 blocks(
        (outW + threads.x - 1) / threads.x,
        (outH + threads.y - 1) / threads.y,
        N * C
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "efficient_avg_pool2d_kernel", ([&] {
        efficient_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &efficient_avg_pool2d_forward, "Optimized 2D Average Pooling forward (CUDA)");
}
