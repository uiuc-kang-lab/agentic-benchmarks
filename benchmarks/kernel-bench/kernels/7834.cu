#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized CUDA kernel for small convolutions
__global__ void conv2d_kernel_small(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_height,
    const int kernel_width,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding) {
    
    extern __shared__ float shared_weight[];
    
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride_size = gridDim.x * blockDim.x;
    
    const int weights_per_thread = (kernel_height * kernel_width * in_channels + blockDim.x - 1) / blockDim.x; // Calculate weights per thread
    for(int i = 0; i < weights_per_thread; i++) {
        const int idx = threadIdx.x * weights_per_thread + i;
        if(idx < kernel_height * kernel_width * in_channels) {
            shared_weight[idx] = weight[idx];
        }
    }
    __syncthreads();
    
    for (int pos = thread_id; pos < batch_size * out_channels * out_height * out_width; pos += stride_size) {
        const int w_out = pos % out_width;
        const int h_out = (pos / out_width) % out_height;
        const int c_out = (pos / (out_width * out_height)) % out_channels;
        const int b = pos / (out_width * out_height * out_channels);
        
        float sum = bias ? bias[c_out] : 0.0f;
        
        #pragma unroll
        for (int c_in = 0; c_in < in_channels; c_in++) {
            #pragma unroll
            for (int kh = 0; kh < kernel_height; kh++) {
                #pragma unroll
                for (int kw = 0; kw < kernel_width; kw++) {
                    const int h_in = h_out * stride - padding + kh;
                    const int w_in = w_out * stride - padding + kw;
                    
                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                        const int input_idx = ((b * in_channels + c_in) * in_height + h_in) * in_width + w_in;
                        const int weight_idx = ((c_out * in_channels + c_in) * kernel_height + kh) * kernel_width + kw;
                        sum += input[input_idx] * shared_weight[weight_idx];
                    }
                }
            }
        }
        output[pos] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }
    
    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto in_height = x.size(2);
    const auto in_width = x.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_height = weight.size(2);
    const auto kernel_width = weight.size(3);
    
    if (kernel_height * kernel_width > 25 || dilation != 1 || groups != 1) {
        return torch::conv2d(x, weight, 
                           bias.has_value() ? bias.value() : torch::Tensor(),
                           {stride, stride}, 
                           {padding, padding}, 
                           {dilation, dilation}, 
                           groups);
    }
    
    const auto out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
    const auto out_width = (in_width + 2 * padding - kernel_width) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, out_height, out_width},
                              x.options());
    
    const int threads = 256;
    const int blocks = (batch_size * out_channels * out_height * out_width + threads - 1) / threads;
    const int shared_memory_size = kernel_height * kernel_width * in_channels * sizeof(float);
    
    conv2d_kernel_small<<<blocks, threads, shared_memory_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_height, in_width,
        kernel_height, kernel_width,
        out_height, out_width,
        stride, padding);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive CUDA 2D Convolution");
}