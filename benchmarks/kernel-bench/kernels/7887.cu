#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_size,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride_step = blockDim.x * gridDim.x;
    
    for (int pos = tid; pos < batch_size * out_channels * output_height * output_width; pos += stride_step) {
        const int w_out = pos % output_width;
        const int h_out = (pos / output_width) % output_height;
        const int c_out = (pos / (output_width * output_height)) % out_channels;
        const int b = pos / (output_width * output_height * out_channels);
        
        float sum = bias ? bias[c_out] : 0.0f;
        
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    const int h_in = h_out * stride - padding + kh;
                    const int w_in = w_out * stride - padding + kw;
                    
                    if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                        const float input_val = input[
                            b * (in_channels * input_height * input_width) +
                            c_in * (input_height * input_width) +
                            h_in * input_width +
                            w_in];
                        const float weight_val = weight[
                            c_out * (in_channels * kernel_size * kernel_size) +
                            c_in * (kernel_size * kernel_size) +
                            kh * kernel_size +
                            kw];
                        sum += input_val * weight_val;
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
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_height = x.size(2);
    const int input_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width},
                              x.options());
    
    const int threads = 256;
    const int blocks = (batch_size * out_channels * output_height * output_width + threads - 1) / threads;
    
    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        output_height,
        output_width,
        stride,
        padding);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with optional bias");
}