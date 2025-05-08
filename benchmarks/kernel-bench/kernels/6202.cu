#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for frequently accessed parameters
__constant__ int c_kernel_size;
__constant__ int c_stride;
__constant__ int c_padding;

__global__ void avg_pool3d_tuned_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w) {
    
    // 2D block configuration for better occupancy
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int idx_x = blockIdx.x * blockDim.x + tid_x;
    const int idx_y = blockIdx.y * blockDim.y + tid_y;
    
    // Calculate batch and channel indices
    const int batch_channel_idx = blockIdx.z;
    const int n = batch_channel_idx / channels;
    const int c = batch_channel_idx % channels;
    
    // Early exit conditions
    if (idx_x >= out_w || idx_y >= out_h || n >= batch_size)
        return;
    
    // Process multiple depth slices per thread block
    for (int d_out = 0; d_out < out_d; d_out++) {
        // Calculate window boundaries
        const int d_start = d_out * c_stride - c_padding;
        const int h_start = idx_y * c_stride - c_padding;
        const int w_start = idx_x * c_stride - c_padding;
        
        // Calculate clamped boundaries
        const int d_start_clamped = max(0, d_start);
        const int h_start_clamped = max(0, h_start);
        const int w_start_clamped = max(0, w_start);
        const int d_end = min(d_start + c_kernel_size, in_d);
        const int h_end = min(h_start + c_kernel_size, in_h);
        const int w_end = min(w_start + c_kernel_size, in_w);
        
        float sum = 0.0f;
        const int base_idx = ((n * channels + c) * in_d);
        
        #pragma unroll 3
        for (int d = d_start_clamped; d < d_end; ++d) {
            const int d_offset = (base_idx + d) * in_h;
            #pragma unroll 3
            for (int h = h_start_clamped; h < h_end; ++h) {
                const int h_offset = (d_offset + h) * in_w;
                #pragma unroll 3
                for (int w = w_start_clamped; w < w_end; ++w) {
                    sum += __ldg(&input[h_offset + w]);
                }
            }
        }
        
        const float pool_size = static_cast<float>(c_kernel_size * c_kernel_size * c_kernel_size);
        const int out_idx = (((n * channels + c) * out_d + d_out) * out_h + idx_y) * out_w + idx_x;
        output[out_idx] = sum / pool_size;
    }
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_d = input.size(2);
    const int in_h = input.size(3);
    const int in_w = input.size(4);
    
    const int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    const int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    const int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    // Copy constants to constant memory
    cudaMemcpyToSymbol(c_kernel_size, &kernel_size, sizeof(int));
    cudaMemcpyToSymbol(c_stride, &stride, sizeof(int));
    cudaMemcpyToSymbol(c_padding, &padding, sizeof(int));
    
    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());
    
    // Optimized block configuration for H100
    dim3 threads(32, 16);  // 512 threads per block
    dim3 blocks(
        (out_w + threads.x - 1) / threads.x,
        (out_h + threads.y - 1) / threads.y,
        batch_size * channels
    );
    
    avg_pool3d_tuned_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward (CUDA)");
}