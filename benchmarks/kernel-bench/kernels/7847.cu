#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16
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
    const int kernel_height,
    const int kernel_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding) {
    
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_ch = blockIdx.z;
    
    if (out_x >= output_width || out_y >= output_height || out_ch >= out_channels)
        return;
        
    float sum = bias ? bias[out_ch] : 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int in_x = out_x * stride - padding + kw;
                    int in_y = out_y * stride - padding + kh;
                    
                    if (in_x >= 0 && in_x < input_width && 
                        in_y >= 0 && in_y < input_height) {
                        
                        float input_val = input[
                            ((b * in_channels + ic) * input_height + in_y) * input_width + in_x];
                        float weight_val = weight[
                            ((out_ch * in_channels + ic) * kernel_height + kh) * kernel_width + kw];
                        sum += input_val * weight_val;
                    }
                }
            }
        }
    }
    
    output[((out_ch * output_height + out_y) * output_width + out_x)] = sum;
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

    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto input_height = x.size(2);
    auto input_width = x.size(3);
    auto out_channels = weight.size(0);
    auto kernel_height = weight.size(2);
    auto kernel_width = weight.size(3);
    
    auto output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    auto output_width = (input_width + 2 * padding - kernel_width) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width},
                              x.options());
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        out_channels
    );
    
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
        kernel_height,
        kernel_width,
        output_height,
        output_width,
        stride,
        padding
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution");
}