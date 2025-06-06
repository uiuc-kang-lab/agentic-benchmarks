#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel that combines manual unrolling and efficient boundary checks

template <typename scalar_t>
__global__ void optimized_avg_pool2d_kernel(
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
    // Map threads in a 2D block to output spatial dimensions
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int out_x = blockIdx.x * blockDim.x + tid_x;
    const int out_y = blockIdx.y * blockDim.y + tid_y;

    // Use blockIdx.z to cover the (N * C) dimension
    const int nc = blockIdx.z;
    const int n = nc / C;
    const int c = nc % C;

    if (out_x >= outW || out_y >= outH)
        return;

    // Calculate the starting coordinates in the input corresponding to the output pixel
    const int in_x_start = out_x * stride - padding;
    const int in_y_start = out_y * stride - padding;
    const int in_x_end = in_x_start + kernel_size;
    const int in_y_end = in_y_start + kernel_size;

    scalar_t sum = scalar_t(0);

    // Fast path: if kernel_size is 3 and the window is fully inside, manually unroll loops
    if (kernel_size == 3 && in_x_start >= 0 && in_y_start >= 0 && in_x_end <= W && in_y_end <= H) {
        int base = (n * C + c) * H;
        int ix = in_x_start;
        int row0 = base + in_y_start;
        int row1 = base + in_y_start + 1;
        int row2 = base + in_y_start + 2;
        sum = input[row0 * W + ix]     + input[row0 * W + ix + 1]     + input[row0 * W + ix + 2] +
              input[row1 * W + ix]     + input[row1 * W + ix + 1]     + input[row1 * W + ix + 2] +
              input[row2 * W + ix]     + input[row2 * W + ix + 1]     + input[row2 * W + ix + 2];
    } else {
        // For border cases, check boundaries per element
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ky++) {
            int y = in_y_start + ky;
            #pragma unroll
            for (int kx = 0; kx < kernel_size; kx++) {
                int x = in_x_start + kx;
                if (y >= 0 && y < H && x >= 0 && x < W) {
                    int offset = ((n * C + c) * H + y) * W + x;
                    sum += input[offset];
                }
            }
        }
    }

    const int out_idx = ((n * C + c) * outH + out_y) * outW + out_x;
    output[out_idx] = sum / static_cast<scalar_t>(kernel_size * kernel_size);
}


// Forward function exposed to PyTorch
torch::Tensor optimized_avg_pool2d_forward(
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
    
    // Calculate output dimensions
    const int outH = (H + 2 * padding - kernel_size) / stride + 1;
    const int outW = (W + 2 * padding - kernel_size) / stride + 1;
    
    auto x_cont = x.contiguous();
    auto options = x.options();
    auto output = torch::empty({N, C, outH, outW}, options);
    
    // Configure 2D thread blocks and grid over spatial outputs and combined N * C in blockIdx.z
    dim3 threads(32, 8);
    dim3 blocks(
        (outW + threads.x - 1) / threads.x,
        (outH + threads.y - 1) / threads.y,
        N * C
    );
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "optimized_avg_pool2d_kernel", ([&] {
        optimized_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &optimized_avg_pool2d_forward, "Optimized 2D Average Pooling forward (CUDA)");
}
