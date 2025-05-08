#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel to perform average pooling using warp shuffle for reduction

// Compute the warp shuffle-based reduction to accumulate results
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void avg_pool2d_forward_kernel_warp_shfl(
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
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (index >= total) return;

    int w_out = index % outW;
    int h_out = (index / outW) % outH;
    int c = (index / (outW * outH)) % C;
    int n = index / (outW * outH * C);

    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    float sum_val = 0.0f;
    int kernel_area = kernel_size * kernel_size;

    // All threads in a warp compute part of the pooling operation
    for (int i = threadIdx.x % 32; i < kernel_area; i += 32) {
        int ki = i / kernel_size;
        int kj = i % kernel_size;

        int h_in = h_start + ki;
        int w_in = w_start + kj;

        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            sum_val += input[((n * C + c) * H + h_in) * W + w_in];
        }
    }

    // Reduce within a warp
    sum_val = warpReduceSum(sum_val);

    // Only the first thread in the warp writes back the result
    if (threadIdx.x % 32 == 0) {
        output[index] = sum_val / kernel_area;
    }
}

// Host function that sets up the kernel launch with improved warp efficiency

torch::Tensor avg_pool2d_forward_warp_shfl(
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

    int outH = (H + 2 * padding - kernel_size) / stride + 1;
    int outW = (W + 2 * padding - kernel_size) / stride + 1;

    auto x_cont = x.contiguous();
    auto options = x.options();
    auto out = torch::empty({N, C, outH, outW}, options);

    const int threads = 256;
    const int blocks = (N * C * outH * outW + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward_kernel_warp_shfl", ([&] {
        const scalar_t* input_data = x_cont.data_ptr<scalar_t>();
        scalar_t* output_data = out.data_ptr<scalar_t>();

        avg_pool2d_forward_kernel_warp_shfl<<<blocks, threads>>>(
            input_data,
            output_data,
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
    m.def("forward", &avg_pool2d_forward_warp_shfl, "2D Average Pooling forward with warp shuffle optimization (CUDA)");
}