#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that minimizes warp divergence by restructuring conditional logic
// for uniform control flow within warps.

template <typename scalar_t>
__global__ void warp_divergence_avg_pool2d_kernel(
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
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc % C;

    if (out_x >= outW || out_y >= outH)
        return;

    int in_x_start = out_x * stride - padding;
    int in_y_start = out_y * stride - padding;
    scalar_t sum = scalar_t(0);

    // Calculate boundaries and clamp to input size
    int h_start = max(0, in_y_start);
    int w_start = max(0, in_x_start);
    int h_end = min(H, in_y_start + kernel_size);
    int w_end = min(W, in_x_start + kernel_size);

    // Use a single path with pre-calculated bounds to avoid warp divergence
    for (int h = h_start; h < h_end; ++h) {
        int row_offset = ((n * C + c) * H + h) * W;
        for (int w = w_start; w < w_end; ++w) {
            sum += input[row_offset + w];
        }
    }

    int out_index = ((n * C + c) * outH + out_y) * outW + out_x;
    output[out_index] = sum / static_cast<scalar_t>(kernel_size * kernel_size);
}

// Forward function exposed to PyTorch

torch::Tensor warp_divergence_avg_pool2d_forward(
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
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "warp_divergence_avg_pool2d_kernel", ([&] {
        warp_divergence_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &warp_divergence_avg_pool2d_forward, "Warp Divergence Minimized 2D Average Pooling forward (CUDA)");
}