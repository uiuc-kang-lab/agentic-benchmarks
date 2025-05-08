#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void avg_pool3d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {
    
    // Calculate position using width-first ordering for better coalescing
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_out = blockIdx.y;
    const int d_out = blockIdx.z % out_d;
    const int c = (blockIdx.z / out_d) % channels;
    const int n = blockIdx.z / (out_d * channels);
    
    if (w_out >= out_w || h_out >= out_h || n >= batch_size)
        return;
    
    // Compute window bounds
    const int d_start = d_out * stride - padding;
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;
    
    const int d_end = min(d_start + kernel_size, in_d);
    const int h_end = min(h_start + kernel_size, in_h);
    const int w_end = min(w_start + kernel_size, in_w);
    
    const int d_start_clamped = max(d_start, 0);
    const int h_start_clamped = max(h_start, 0);
    const int w_start_clamped = max(w_start, 0);
    
    float sum = 0.0f;
    const int base_idx = ((n * channels + c) * in_d);
    
    #pragma unroll
    for (int d = d_start_clamped; d < d_end; ++d) {
        const int d_idx = (base_idx + d) * in_h;
        #pragma unroll
        for (int h = h_start_clamped; h < h_end; ++h) {
            const int h_idx = (d_idx + h) * in_w;
            #pragma unroll
            for (int w = w_start_clamped; w < w_end; ++w) {
                sum += input[h_idx + w];
            }
        }
    }
    
    const int pool_volume = kernel_size * kernel_size * kernel_size;
    const int out_idx = (((n * channels + c) * out_d + d_out) * out_h + h_out) * out_w + w_out;
    output[out_idx] = sum / static_cast<float>(pool_volume);
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
    
    dim3 threads(256, 1, 1);
    dim3 blocks(
        (out_w + threads.x - 1) / threads.x,
        out_h,
        batch_size * channels * out_d
    );
    
    avg_pool3d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward (CUDA)");
}