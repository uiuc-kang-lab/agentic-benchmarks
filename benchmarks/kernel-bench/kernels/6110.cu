#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Store frequently accessed read-only offsets for 3x3 pooling in constant memory
// These arrays are small and remain the same for every kernel invocation
__constant__ int pool3_dx[3] = {0, 1, 2};
__constant__ int pool3_dy[3] = {0, 1, 2};

// Kernel that uses constant memory for the fast 3x3 pooling case

template <typename scalar_t>
__global__ void constant_optimized_avg_pool2d_kernel(
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
    // Determine spatial position in the output tensor
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + tid_x;
    int out_y = blockIdx.y * blockDim.y + tid_y;

    // Use blockIdx.z to cover the (N * C) dimension
    int nc = blockIdx.z;
    if(nc >= N * C) return;
    int n = nc / C;
    int c = nc % C;

    // Check if the computed spatial indices are within output bounds
    if (out_x >= outW || out_y >= outH)
        return;

    // Compute the starting location in the input corresponding to the output element
    int in_x_start = out_x * stride - padding;
    int in_y_start = out_y * stride - padding;

    scalar_t sum = static_cast<scalar_t>(0);

    // Fast path: use constant memory for 3x3 pooling when the window is fully inside the input
    if (kernel_size == 3 && in_x_start >= 0 && in_y_start >= 0 && 
        (in_x_start + 3) <= W && (in_y_start + 3) <= H) {
        // Compute the base offset for the (n, c) slice
        int base = (n * C + c) * H;
        int row = in_y_start;
        int col = in_x_start;
        sum = input[(base + row + pool3_dy[0]) * W + (col + pool3_dx[0])] +
              input[(base + row + pool3_dy[0]) * W + (col + pool3_dx[1])] +
              input[(base + row + pool3_dy[0]) * W + (col + pool3_dx[2])] +
              input[(base + row + pool3_dy[1]) * W + (col + pool3_dx[0])] +
              input[(base + row + pool3_dy[1]) * W + (col + pool3_dx[1])] +
              input[(base + row + pool3_dy[1]) * W + (col + pool3_dx[2])] +
              input[(base + row + pool3_dy[2]) * W + (col + pool3_dx[0])] +
              input[(base + row + pool3_dy[2]) * W + (col + pool3_dx[1])] +
              input[(base + row + pool3_dy[2]) * W + (col + pool3_dx[2])];
    } else {
        // Generic path with boundary checks
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
    
    // Write result to output and normalize by window area
    int out_index = ((n * C + c) * outH + out_y) * outW + out_x;
    output[out_index] = sum / static_cast<scalar_t>(kernel_size * kernel_size);
}

// Forward function exposed to PyTorch

torch::Tensor constant_optimized_avg_pool2d_forward(
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

    // Calculate output dimensions using standard pooling formula
    const int outH = (H + 2 * padding - kernel_size) / stride + 1;
    const int outW = (W + 2 * padding - kernel_size) / stride + 1;

    auto x_contiguous = x.contiguous();
    auto output = torch::empty({N, C, outH, outW}, x.options());

    // Configure 2D thread blocks over spatial dimensions with blockIdx.z covering N * C
    dim3 threads(32, 8);
    dim3 blocks(
        (outW + threads.x - 1) / threads.x,
        (outH + threads.y - 1) / threads.y,
        N * C
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "constant_optimized_avg_pool2d_kernel", ([&] {
        constant_optimized_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            x_contiguous.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C, H, W, outH, outW,
            kernel_size, stride, padding
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &constant_optimized_avg_pool2d_forward, "Constant Memory Optimized 2D Average Pooling forward (CUDA)");
}
