#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined 2D average pooling kernel that uses a fast-path when the pooling window is completely in-bound
// and falls back to a safe path when on the borders.

template <typename scalar_t>
__global__ void avg_pool2d_forward_kernel_combined(
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

    // Compute output coordinates
    int w_out = index % outW;
    int h_out = (index / outW) % outH;
    int c = (index / (outW * outH)) % C;
    int n = index / (outW * outH * C);

    // Start indices in input tensor
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    scalar_t sum_val = scalar_t(0);
    
    // Fast path: if the entire pooling window is within bounds, no need for per-element boundary checks.
    bool fast_path = (h_start >= 0) && ((h_start + kernel_size) <= H) &&
                     (w_start >= 0) && ((w_start + kernel_size) <= W);

    if (fast_path) {
        #pragma unroll
        for (int i = 0; i < kernel_size; i++) {
            #pragma unroll
            for (int j = 0; j < kernel_size; j++) {
                int h_in = h_start + i;
                int w_in = w_start + j;
                sum_val += input[((n * C + c) * H + h_in) * W + w_in];
            }
        }
    } else {
        // Safe path for border cases
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                int h_in = h_start + i;
                int w_in = w_start + j;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    sum_val += input[((n * C + c) * H + h_in) * W + w_in];
                }
            }
        }
    }
    
    // Compute the average and write output
    output[index] = sum_val / static_cast<scalar_t>(kernel_size * kernel_size);
}

// Host function that sets up the kernel launch

torch::Tensor avg_pool2d_forward_combined(
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

    // Use 2D block configuration for better memory access patterns
    const dim3 threads(16, 16);  // 256 threads total, arranged in 2D
    const dim3 blocks(
        (outW + threads.x - 1) / threads.x,
        (outH + threads.y - 1) / threads.y,
        N * C  // One block-layer per image-channel combination
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward_kernel_combined", ([&] {
        const scalar_t* input_data = x_cont.data_ptr<scalar_t>();
        scalar_t* output_data = out.data_ptr<scalar_t>();

        avg_pool2d_forward_kernel_combined<<<blocks, threads>>>(
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
    m.def("forward", &avg_pool2d_forward_combined, "2D Average Pooling forward combined optimized kernel (CUDA)");
}
