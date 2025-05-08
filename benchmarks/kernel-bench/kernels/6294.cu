#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized with precomputed reciprocal and grid optimization
__global__ void avg_pool3d_forward_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding,
    float inv_pool_volume) {

    // Use registers more efficiently by combining dimensions
    const int out_hw = out_h * out_w;
    const int out_dhw = out_d * out_hw;
    const int total_elements = batch_size * channels * out_dhw;
    
    // Grid-stride loop
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total_elements;
         index += blockDim.x * gridDim.x) {
         
        // More efficient index calculation using combined dimensions
        const int idx_w = index % out_w;
        const int idx_h = (index / out_w) % out_h;
        const int idx_d = (index / out_hw) % out_d;
        const int idx_c = (index / out_dhw) % channels;
        const int idx_n = index / (out_dhw * channels);

        // Calculate window boundaries
        const int d_start = idx_d * stride - padding;
        const int h_start = idx_h * stride - padding;
        const int w_start = idx_w * stride - padding;

        // Calculate window ends
        const int d_end = d_start + kernel_size;
        const int h_end = h_start + kernel_size;
        const int w_end = w_start + kernel_size;

        // Clamp boundaries
        const int d_clamped_start = max(0, d_start);
        const int h_clamped_start = max(0, h_start);
        const int w_clamped_start = max(0, w_start);
        const int d_clamped_end = min(in_d, d_end);
        const int h_clamped_end = min(in_h, h_end);
        const int w_clamped_end = min(in_w, w_end);

        float sum = 0.0f;
        for (int d = d_clamped_start; d < d_clamped_end; ++d) {
            for (int h = h_clamped_start; h < h_clamped_end; ++h) {
                for (int w = w_clamped_start; w < w_clamped_end; ++w) {
                    sum += input[((((n * channels) + c) * in_d + d) * in_h + h) * in_w + w];
                }
            }
        }
        output[index] = sum * inv_pool_volume;
    }
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto in_d = input.size(2);
    const auto in_h = input.size(3);
    const auto in_w = input.size(4);

    const int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    const int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    const int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());
    const int total_elements = batch_size * channels * out_d * out_h * out_w;
    const float inv_pool_volume = 1.0f / (kernel_size * kernel_size * kernel_size);

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    avg_pool3d_forward_optimized<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding,
        inv_pool_volume);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel execution failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 3D Average Pooling");
}