#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_3x3_pool(
    const scalar_t* __restrict__ input,
    const int base_offset,
    const int W,
    const int in_x_start
) {
    const int row0 = base_offset;
    const int row1 = row0 + W;
    const int row2 = row1 + W;
    
    return input[row0 + in_x_start]     + input[row0 + in_x_start + 1]     + input[row0 + in_x_start + 2] +
           input[row1 + in_x_start]     + input[row1 + in_x_start + 1]     + input[row1 + in_x_start + 2] +
           input[row2 + in_x_start]     + input[row2 + in_x_start + 1]     + input[row2 + in_x_start + 2];
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_generic_pool(
    const scalar_t* __restrict__ input,
    const int base_offset,
    const int W,
    const int H,
    const int in_x_start,
    const int in_y_start,
    const int kernel_size
) {
    scalar_t sum = 0;
    #pragma unroll
    for (int ky = 0; ky < 7; ++ky) {  // Unroll up to common kernel sizes
        if (ky >= kernel_size) break;
        const int y = in_y_start + ky;
        if (y >= 0 && y < H) {
            const int row_offset = base_offset + y * W;
            #pragma unroll
            for (int kx = 0; kx < 7; ++kx) {
                if (kx >= kernel_size) break;
                const int x = in_x_start + kx;
                if (x >= 0 && x < W) {
                    sum += input[row_offset + x];
                }
            }
        }
    }
    return sum;
}

template <typename scalar_t>
__device__ __forceinline__ bool is_fully_inside(
    const int in_x_start,
    const int in_y_start,
    const int kernel_size,
    const int H,
    const int W
) {
    return (in_x_start >= 0) && (in_y_start >= 0) &&
           (in_x_start + kernel_size <= W) && (in_y_start + kernel_size <= H);
}

template <typename scalar_t>
__global__ void modular_avg_pool2d_kernel(
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
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * outH * outW;
    
    if (tid >= total) return;

    // Calculate output position
    const int w_out = tid % outW;
    const int h_out = (tid / outW) % outH;
    const int c = (tid / (outW * outH)) % C;
    const int n = tid / (outW * outH * C);

    // Calculate input window position
    const int in_x_start = w_out * stride - padding;
    const int in_y_start = h_out * stride - padding;
    
    // Calculate base offset for current batch and channel
    const int base_offset = ((n * C + c) * H) * W;
    
    scalar_t sum;
    
    if (kernel_size == 3 && is_fully_inside<scalar_t>(in_x_start, in_y_start, 3, H, W)) {
        // Optimized path for 3x3 pooling
        sum = compute_3x3_pool<scalar_t>(
            input,
            base_offset + in_y_start * W,
            W,
            in_x_start
        );
    } else {
        // Generic path for other cases
        sum = compute_generic_pool<scalar_t>(
            input,
            base_offset,
            W, H,
            in_x_start,
            in_y_start,
            kernel_size
        );
    }
    
    output[tid] = sum / static_cast<scalar_t>(kernel_size * kernel_size);
}

torch::Tensor modular_avg_pool2d_forward(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding
) {
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor");
    
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    
    const int outH = (H + 2 * padding - kernel_size) / stride + 1;
    const int outW = (W + 2 * padding - kernel_size) / stride + 1;
    
    auto x_cont = x.contiguous();
    auto output = torch::empty({N, C, outH, outW}, x.options());
    
    const int total_elements = N * C * outH * outW;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "modular_avg_pool2d_kernel", ([&] {
        modular_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &modular_avg_pool2d_forward, "Modular 2D Average Pooling forward (CUDA)");
}