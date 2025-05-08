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
    const int in_height,
    const int in_width,
    const int kernel_height,
    const int kernel_width,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding) {
    
    const int n = blockIdx.x;
    const int c_out = blockIdx.y;
    const int h_out = threadIdx.y;
    const int w_out = threadIdx.x;

    if (h_out < out_height && w_out < out_width) {
        float sum = bias ? bias[c_out] : 0.0f;
        
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    const int h_in = h_out * stride - padding + kh;
                    const int w_in = w_out * stride - padding + kw;
                    
                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                        const int input_idx = ((n * in_channels + c_in) * in_height + h_in) * in_width + w_in;
                        const int weight_idx = ((c_out * in_channels + c_in) * kernel_height + kh) * kernel_width + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        const int output_idx = ((n * out_channels + c_out) * out_height + h_out) * out_width + w_out;
        output[output_idx] = sum;
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
    
    if (dilation != 1 || groups != 1) {
        return torch::conv2d(x, weight, bias.has_value() ? bias.value() : torch::Tensor(),
                           {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
    }
    
    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto in_height = x.size(2);
    const auto in_width = x.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_height = weight.size(2);
    const auto kernel_width = weight.size(3);
    
    const auto out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
    const auto out_width = (in_width + 2 * padding - kernel_width) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, out_height, out_width},
                              x.options());
    
    dim3 threads(out_width, out_height);
    dim3 blocks(batch_size, out_channels);
    
    conv2d_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "CUDA 2D Convolution");
}