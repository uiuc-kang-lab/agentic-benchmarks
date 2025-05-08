#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool2d_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernel_size, int stride, int padding
) {
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc % C;
    
    int h_out = blockIdx.x * blockDim.y + threadIdx.y;
    int w_out = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (h_out >= outH || w_out >= outW) return;

    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;
    
    // Pre-compute valid ranges to avoid per-element boundary checks
    int h_start_valid = max(0, h_start);
    int w_start_valid = max(0, w_start);
    int h_end_valid = min(h_start + kernel_size, H);
    int w_end_valid = min(w_start + kernel_size, W);
    
    scalar_t sum_val = 0;
    int valid_elements = (h_end_valid - h_start_valid) * (w_end_valid - w_start_valid);
    
    // Direct indexing without boundary checks within the valid range
    #pragma unroll
    for (int h = h_start_valid; h < h_end_valid; h++) {
        #pragma unroll
        for (int w = w_start_valid; w < w_end_valid; w++) {
            sum_val += input[((n * C + c) * H + h) * W + w];
        }
    }
    
    // Normalize by actual number of valid elements
    output[((n * C + c) * outH + h_out) * outW + w_out] = 
        sum_val / static_cast<scalar_t>(valid_elements);
}

torch::Tensor avg_pool2d_forward(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding
) {
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor.");
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    int outH = (H + 2 * padding - kernel_size)/stride + 1;
    int outW = (W + 2 * padding - kernel_size)/stride + 1;

    auto x_cont = x.contiguous();
    auto out = torch::empty({N, C, outH, outW}, x.options());

    dim3 block(32, 4);
    dim3 grid(
        (outH + block.y - 1) / block.y,
        (outW + block.x - 1) / block.x,
        N * C
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool_forward", ([&] {
        avg_pool2d_forward_kernel<scalar_t><<<grid, block>>>(
            x_cont.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            N, C, H, W,
            outH, outW,
            kernel_size, stride, padding
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool2d_forward, "2D Average Pooling forward (CUDA)");
}