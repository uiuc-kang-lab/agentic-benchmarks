#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void grid_optimized_avg_pool2d_kernel(
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
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    
    const int out_x = blockIdx.x * blockDim.x + tid_x;
    const int out_y = blockIdx.y * blockDim.y + tid_y;
    
    const int nc_idx = blockIdx.z;
    const int n = nc_idx / C;
    const int c = nc_idx % C;

    if (out_x >= outW || out_y >= outH) return;

    const int in_x_start = out_x * stride - padding;
    const int in_y_start = out_y * stride - padding;

    scalar_t sum = 0;

    #pragma unroll
    for (int ky = 0; ky < kernel_size; ky++) {
        const int in_y = in_y_start + ky;
        #pragma unroll
        for (int kx = 0; kx < kernel_size; kx++) {
            const int in_x = in_x_start + kx;
            
            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                sum += input[((n * C + c) * H + in_y) * W + in_x];
            }
        }
    }

    const int out_idx = ((n * C + c) * outH + out_y) * outW + out_x;
    output[out_idx] = sum / static_cast<scalar_t>(kernel_size * kernel_size);
}

torch::Tensor avg_pool2d_forward(
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

    dim3 threads(32, 8);
    dim3 blocks(
        (outW + threads.x - 1) / threads.x,
        (outH + threads.y - 1) / threads.y,
        N * C
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "grid_optimized_avg_pool2d_kernel", ([&] {
        grid_optimized_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &avg_pool2d_forward, "Grid Optimized 2D Average Pooling forward (CUDA)");
}