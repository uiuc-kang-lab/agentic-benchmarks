#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv_transpose1d_kernel(
    const float* input,
    const float* weight,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    if (idx < batch_size * out_channels * output_width) {
        const int w = idx % output_width;
        const int c = (idx / output_width) % out_channels;
        const int b = idx / (output_width * out_channels);
        
        float sum = 0.0f;
        
        #pragma unroll 8
        for (int ic = 0; ic < in_channels; ++ic) {
            const int input_start = w - kernel_size + 1 + padding;
            const int input_end = input_start + kernel_size;
            
            #pragma unroll
            for (int k = 0; k < kernel_size; ++k) {
                const int input_w = (input_start + k) / stride;
                if (input_w >= 0 && input_w < input_width && (input_start + k) % stride == 0) {
                    const int input_idx = b * in_channels * input_width + ic * input_width + input_w;
                    const int weight_idx = c * in_channels * kernel_size + ic * kernel_size + k;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_width = x.size(2);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, output_width},
                              x.options());
    
    const int threads = 256;
    const int blocks = (batch_size * out_channels * output_width + threads - 1) / threads;
    
    conv_transpose1d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_width,
        kernel_size,
        stride,
        padding,
        output_padding);
    
    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1}));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 1D convolution forward (CUDA)");
}