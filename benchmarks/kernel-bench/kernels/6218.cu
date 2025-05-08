#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void avg_pool3d_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_size = blockDim.x * gridDim.x;
    int total_elements = batch_size * channels * out_d * out_h * out_w;
    int pool_volume = kernel_size * kernel_size * kernel_size;

    #pragma unroll 1
    for (; index < total_elements; index += stride_size) {
        // Decompose linear index into output coordinates
        int w_out = index % out_w;
        int tmp = index / out_w;
        int h_out = tmp % out_h;
        tmp = tmp / out_h;
        int d_out = tmp % out_d;
        tmp = tmp / out_d;
        int c = tmp % channels;
        int n = tmp / channels;

        // Calculate window boundaries
        int d_start = d_out * stride - padding;
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;

        // Calculate window endpoints
        int d_end = d_start + kernel_size;
        int h_end = h_start + kernel_size;
        int w_end = w_start + kernel_size;

        // Compute clamped boundaries
        int d0 = max(d_start, 0);
        int h0 = max(h_start, 0);
        int w0 = max(w_start, 0);
        int d1 = min(d_end, in_d);
        int h1 = min(h_end, in_h);
        int w1 = min(w_end, in_w);

        float sum = 0.0f;
        
        // Pre-calculate base offset for channel and batch
        int batch_channel_offset = (n * channels + c) * in_d;
        
        #pragma unroll 1
        for (int d = d0; d < d1; ++d) {
            int depth_offset = (batch_channel_offset + d) * in_h;
            #pragma unroll 1
            for (int h = h0; h < h1; ++h) {
                int height_offset = (depth_offset + h) * in_w;
                #pragma unroll 1
                for (int w = w0; w < w1; ++w) {
                    sum += input[height_offset + w];
                }
            }
        }
        
        output[index] = sum / static_cast<float>(pool_volume);
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
    
    // Use larger thread block size of 512
    const int threads = 512;
    int blocks = (total_elements + threads - 1) / threads;
    
    // Limit maximum number of blocks to avoid overhead
    blocks = min(blocks, 65535);
    
    avg_pool3d_optimized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding);
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward (CUDA)");
}