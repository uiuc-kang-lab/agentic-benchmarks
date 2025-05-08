#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses stride loops to cover workloads larger than the number of available threads,
// ensuring correct boundary handling for 2D average pooling.

template <typename scalar_t>
__global__ void stride_loop_avg_pool2d_kernel(
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
    // Total number of output elements
    const int total = N * C * outH * outW;
    
    // Compute global thread index and grid stride
    const int grid_stride = blockDim.x * gridDim.x;
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += grid_stride) {
        // Decompose 1D index into 4D indices: [n, c, h, w]
        int tmp = index;
        const int w_out = tmp % outW;
        tmp /= outW;
        const int h_out = tmp % outH;
        tmp /= outH;
        const int c = tmp % C;
        const int n = tmp / C;

        // Calculate the top-left corner of the corresponding pooling window in the input
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;

        // Compute pooling window boundaries before clamping
        int h_end = h_start + kernel_size;
        int w_end = w_start + kernel_size;

        // Clamp the pooling window to input boundaries for correct edge handling
        h_start = (h_start < 0) ? 0 : h_start;
        w_start = (w_start < 0) ? 0 : w_start;
        h_end = (h_end > H) ? H : h_end;
        w_end = (w_end > W) ? W : w_end;

        scalar_t sum_val = scalar_t(0);
        
        // Sum over the pooling window
        for (int h = h_start; h < h_end; ++h) {
            int row_offset = ((n * C + c) * H + h) * W;
            for (int w = w_start; w < w_end; ++w) {
                sum_val += input[row_offset + w];
            }
        }

        // Write the averaged result for this output element
        output[index] = sum_val / static_cast<scalar_t>(kernel_size * kernel_size);
    }
}

// Host function to launch the stride loop kernel

torch::Tensor stride_loop_avg_pool2d_forward(
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
    
    // Compute output dimensions
    const int outH = (H + 2 * padding - kernel_size) / stride + 1;
    const int outW = (W + 2 * padding - kernel_size) / stride + 1;
    
    auto x_cont = x.contiguous();
    auto output = torch::empty({N, C, outH, outW}, x.options());
    
    // Total number of output elements
    const int total = N * C * outH * outW;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "stride_loop_avg_pool2d_kernel", ([&] {
        stride_loop_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &stride_loop_avg_pool2d_forward, "Stride Loop 2D Average Pooling forward (CUDA)");
}
