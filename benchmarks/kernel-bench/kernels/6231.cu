#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp-optimized 3D Average Pooling Kernel
__global__ void warp_optimized_avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    const unsigned int FULL_MASK = 0xffffffff;
    const int warpSize = 32;
    const int warpIndex = threadIdx.x & (warpSize - 1);
    const int warpID = threadIdx.x / warpSize;
    
    int index = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int total_elements = batch_size * channels * out_d * out_h * out_w;
    const float inv_pool_volume = 1.0f / (kernel_size * kernel_size * kernel_size);

    while (index < total_elements) {
        // Decompose index into output coordinates
        int w_out = index % out_w;
        int tmp = index / out_w;
        int h_out = tmp % out_h;
        tmp /= out_h;
        int d_out = tmp % out_d;
        tmp /= out_d;
        int c = tmp % channels;
        int n = tmp / channels;

        // Calculate window boundaries
        int d_start = d_out * stride - padding;
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;

        int d0 = max(d_start, 0);
        int h0 = max(h_start, 0);
        int w0 = max(w_start, 0);
        int d1 = min(d_start + kernel_size, in_d);
        int h1 = min(h_start + kernel_size, in_h);
        int w1 = min(w_start + kernel_size, in_w);

        float local_sum = 0.0f;
        // Distribute work across warp threads
        int base_nc = (n * channels + c) * in_d * in_h * in_w;
        
        // Each thread in warp processes different elements in the pooling window
        for (int d = d0; d < d1; ++d) {
            int base_d = base_nc + d * in_h * in_w;
            for (int h = h0; h < h1; ++h) {
                int base_h = base_d + h * in_w;
                for (int w = w0 + warpIndex; w < w1; w += warpSize) {
                    local_sum += input[base_h + w];
                }
            }
        }

        // Warp-level reduction using shuffle operations
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
        }

        // First thread in warp writes the result
        if (warpIndex == 0) {
            output[index] = local_sum * inv_pool_volume;
        }

        index += (gridDim.x * blockDim.x) / warpSize;
    }
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);

    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    int total_elements = batch_size * channels * out_d * out_h * out_w;
    const int threads = 256; // Multiple of warp size (32)
    int blocks = (total_elements + (threads / 32) - 1) / (threads / 32);

    warp_optimized_avg_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) warp-optimized version");
}