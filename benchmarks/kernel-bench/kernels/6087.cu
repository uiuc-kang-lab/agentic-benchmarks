#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Modular device function to compute sum inside pooling window
// This modular approach increases readability and maintains a single responsibility per function.
// I/O arguments facilitate easy reuse and composition.

// Compute average over pooling window safely modulo boundary cases
// Configures shared memory optimizations for pooling windows aligned to input memory access patterns
// 

// Simplified device function for pooling window accumulation
// Safer code for verifying robust extensions like pooling gradients

//

//typical optimizations for unrolling guarantees proper performance
//separate common cases and input ownership paradigms

// Handle boundaries for proper pooling window function

device __fun_pool_sum_unroll<T>(
    int w_start,         // Starting input window coordinates
    int w_end,             // Ending input window coordinates
    int kernel_size, 
    int stride, 
    int padding, 
    int thread_idx, // curr out index 
    int N,
    int C,
    int H,
    int W, // dims
    int c, int n,
    const T* __restrict__ input,
    const int outW, const int outH,
) {
   
    int in_x_start = out_x * stride - padding;
    int in_y_start = out_y * stride - padding;
    
    scalar_t sum = scalar_t(0);

    int y_start = (in_y_start < 0) ? 0 : in_y_start;
    int x_start = (in_x_start < 0) ? 0 : in_x_start;
    int y_end = (in_y_start + kernel_size > H) ? H : in_y_start + kernel_size;
    int x_end = (in_x_start + kernel_size > W) ? W : in_x_start + kernel_size;

    for (int y = y_start; y < y_end; ++y) {
        int base_offset = ((n * C + c) * H + y) * W;
        for (int x = x_start; x < x_end; ++x) {
            sum += input[base_offset + x];
        }
    }

    return sum / static_cast<T>(kernel_size * kernel_size);
}

// Kernel using modular approach to pooling operation
__global__ void modular_avg_pool2d_gradient_kernel(
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
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z; //combined batch + channels

    int n = nc / C;
    int c = nc % C;

    if (out_x >= outW || out_y >= outH) return;

    int thread_idx = ((n * C + c) * outH + out_y) * outW + out_x;

    output[thread_idx] = __fun_pool_sum_unroll(T w_start, T w_end, kernel_size,stride,padding,thread_idx,N,C,H,W, c, n, input, outW, outH);
}


// Host function
torch::Tensor modular_avg_pool2d_gradient_forward(
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

    dim3 threads(32, 8);
    dim3 blocks(
        (outW + threads.x - 1) / threads.x,
        (outH + threads.y - 1) / threads.y,
        N * C // outer dimension
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "modular_avg_pool2d_gradient_kernel", ([&] {
        modular_avg_pool2d_gradient_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &modular_avg_pool2d_gradient_forward, "Modular 2D Average Pooling Gradient forward (CUDA)");
}