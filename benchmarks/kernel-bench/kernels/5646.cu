#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__device__ __forceinline__ int compute_output_dim(int input_dim, int padding, int dilation, int kernel_size, int stride) {
    return (input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t pooled_max(
    const scalar_t* __restrict__ input,
    const int base_offset,
    const int ih_start,
    const int iw_start,
    const int kernel_size,
    const int dilation,
    const int input_height,
    const int input_width,
    const int stride
) {
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    #pragma unroll
    for (int kh = 0; kh < kernel_size; ++kh) {
        int ih = ih_start + kh * dilation;
        if (ih < 0 || ih >= input_height) continue;
        
        #pragma unroll
        for (int kw = 0; kw < kernel_size; ++kw) {
            int iw = iw_start + kw * dilation;
            if (iw < 0 || iw >= input_width) continue;
            
            scalar_t val = __ldg(&input[base_offset + ih * input_width + iw]);
            max_val = max(max_val, val);
        }
    }
    return max_val;
}

template <typename scalar_t>
__global__ void maxpool2d_optimized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    const int total = batch_size * channels * output_height * output_width;
    const int grid_stride = blockDim.x * gridDim.x;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total; 
         idx += grid_stride) {
        
        const int ow = idx % output_width;
        const int oh = (idx / output_width) % output_height;
        const int c = (idx / (output_width * output_height)) % channels;
        const int b = idx / (output_width * output_height * channels);
        
        const int base_offset = (b * channels + c) * input_height * input_width;
        const int ih_start = oh * stride - padding;
        const int iw_start = ow * stride - padding;
        
        output[idx] = pooled_max<scalar_t>(
            input, base_offset,
            ih_start, iw_start,
            kernel_size, dilation,
            input_height, input_width, stride
        );
    }
}

torch::Tensor maxpool2d_cuda_optimized_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int out_h = compute_output_dim<int>(height, padding, dilation, kernel_size, stride);
    const int out_w = compute_output_dim<int>(width, padding, dilation, kernel_size, stride);
    
    auto output = torch::empty({batch, channels, out_h, out_w}, input.options());
    
    const int threads = 256;
    const int elements = batch * channels * out_h * out_w;
    const int blocks = (elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "maxpool2d_forward", ([&] {
        maxpool2d_optimized_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch, channels,
            height, width,
            out_h, out_w,
            kernel_size,
            stride, padding, dilation
        );
    }));
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &maxpool2d_cuda_optimized_forward, "Optimized MaxPool2D forward (CUDA)");
}