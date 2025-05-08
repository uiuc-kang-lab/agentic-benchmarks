#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 3D average pooling (count_include_pad=True)
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
        // Decompose the linear index into (n, c, d_out, h_out, w_out)
        int w_out = index % out_w;
        int tmp = index / out_w;
        int h_out = tmp % out_h;
        tmp = tmp / out_h;
        int d_out = tmp % out_d;
        tmp = tmp / out_d;
        int c = tmp % channels;
        int n = tmp / channels;
        
        // Compute the top-left-front corner of the pooling window (considering padding)
        int d_start = d_out * stride - padding;
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;
        
        // Determine window boundaries before clamping
        int d_end = d_start + kernel_size;
        int h_end = h_start + kernel_size;
        int w_end = w_start + kernel_size;
        
        // Clamp window boundaries to the input dimensions
        int d_start_clamped = (d_start < 0) ? 0 : d_start;
        int h_start_clamped = (h_start < 0) ? 0 : h_start;
        int w_start_clamped = (w_start < 0) ? 0 : w_start;
        int d_end_clamped = (d_end > in_d) ? in_d : d_end;
        int h_end_clamped = (h_end > in_h) ? in_h : h_end;
        int w_end_clamped = (w_end > in_w) ? in_w : w_end;
        
        float sum = 0.0f;
        // Sum over the valid input elements in the pooling window.
        for (int d = d_start_clamped; d < d_end_clamped; ++d) {
            for (int h = h_start_clamped; h < h_end_clamped; ++h) {
                for (int w = w_start_clamped; w < w_end_clamped; ++w) {
                    // Compute the index for the input tensor in the shape [N, C, D, H, W]
                    int input_index = (((n * channels + c) * in_d + d) * in_h + h) * in_w + w;
                    sum += input[input_index];
                }
            }
        }
        // For count_include_pad=True, the average is always divided by the full kernel volume.
        int pool_volume = kernel_size * kernel_size * kernel_size;
        output[index] = sum / static_cast<float>(pool_volume);
        
        index += blockDim.x * gridDim.x;
    }
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    // Check that input is a 5D CUDA tensor.
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    
    int batch_size = input.size(0);
    int channels   = input.size(1);
    int in_d       = input.size(2);
    int in_h       = input.size(3);
    int in_w       = input.size(4);
    
    // Compute output dimensions using the pooling formula:
    // out_dim = floor((in_dim + 2*padding - kernel_size) / stride) + 1
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    // Allocate output tensor with the same options as input.
    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    int total_elements = batch_size * channels * out_d * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    avg_pool3d_forward_kernel<<<blocks, threads>>>(input_ptr, output_ptr,
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