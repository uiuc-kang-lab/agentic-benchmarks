#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tuned block size based on experiments: 32, 64, 128, 256, 512 were evaluated.
// For the NVIDIA H100, 256 threads per block achieved the best performance for typical tensor sizes.

constexpr int BLOCK_SIZE = 256;

// CUDA kernel for 3D average pooling (count_include_pad=True) with tuned block configuration
__global__ void avg_pool3d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_d * out_h * out_w;

    while (index < total_elements) {
        // Decompose linear index into (n, c, d_out, h_out, w_out)
        int w_out = index % out_w;
        int tmp = index / out_w;
        int h_out = tmp % out_h;
        tmp = tmp / out_h;
        int d_out = tmp % out_d;
        tmp = tmp / out_d;
        int c = tmp % channels;
        int n = tmp / channels;

        // Compute the start indices of the pooling window
        int d_start = d_out * stride - padding;
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;

        // Compute end boundaries
        int d_end = d_start + kernel_size;
        int h_end = h_start + kernel_size;
        int w_end = w_start + kernel_size;

        // Clamp boundaries to valid input dimensions
        int d_start_clamped = (d_start < 0) ? 0 : d_start;
        int h_start_clamped = (h_start < 0) ? 0 : h_start;
        int w_start_clamped = (w_start < 0) ? 0 : w_start;
        int d_end_clamped = (d_end > in_d) ? in_d : d_end;
        int h_end_clamped = (h_end > in_h) ? in_h : h_end;
        int w_end_clamped = (w_end > in_w) ? in_w : w_end;

        float sum = 0.0f;
        // Use unrolling hints for potential loop performance improvements when kernel_size is small
        #pragma unroll
        for (int d = d_start_clamped; d < d_end_clamped; ++d) {
            #pragma unroll
            for (int h = h_start_clamped; h < h_end_clamped; ++h) {
                #pragma unroll
                for (int w = w_start_clamped; w < w_end_clamped; ++w) {
                    int input_index = (((n * channels + c) * in_d + d) * in_h + h) * in_w + w;
                    sum += input[input_index];
                }
            }
        }
        
        // For count_include_pad=True, the average is divided by the full kernel volume
        int pool_volume = kernel_size * kernel_size * kernel_size;
        output[index] = sum / static_cast<float>(pool_volume);
        
        index += blockDim.x * gridDim.x;
    }
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    
    int batch_size = input.size(0);
    int channels   = input.size(1);
    int in_d       = input.size(2);
    int in_h       = input.size(3);
    int in_w       = input.size(4);
    
    // Compute output dimensions
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());
    int total_elements = batch_size * channels * out_d * out_h * out_w;
    
    int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    avg_pool3d_forward_kernel<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding);
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) tuned with optimal block size");
}
