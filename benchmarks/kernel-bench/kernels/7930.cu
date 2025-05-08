#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_kernel(
float* output, const float* input, const float* weight,
const float* bias, int batch_size, int in_channels,
int out_channels, int height, int width,
int kernel_size, int stride, int padding) {
    int b = blockIdx.x;
    int oc = blockIdx.y;
    int h = blockIdx.z / output_width;
    int w = blockIdx.z % ((width + stride - 1) / stride);
    
    if (b >= batch_size || oc >= out_channels ||
        h * stride >= height || w * stride >= width) return;
    
    float sum = bias ? bias[oc] : 0.0f;
    
    #pragma unroll 3
    for (int ic = 0; ic < in_channels; ic++) {
        #pragma unroll 3
        for (int kh = 0; kh < kernel_size; kh++) {
            #pragma unroll 3
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h * stride - padding + kh;
                int w_in = w * stride - padding + kw;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    float input_val = input[((b * in_channels + ic) * height + h_in) * width + w_in];
                    float weight_val = weight[((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw];
                    sum += input_val * weight_val;
                }
            }
        }
    }
    
    int out_idx = ((b * out_channels + oc) * height + h) * width + w;
    output[out_idx] = sum;
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
    auto height = x.size(2);
    auto width = x.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    auto output_height = (height + 2 * padding - kernel_size) / stride + 1;
    auto output_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width},
                              x.options());
    
    dim3 threads(1);
    dim3 blocks(batch_size, out_channels, output_height * output_width);
    
    conv2d_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        batch_size, in_channels, out_channels,
        height, width, kernel_size,
        stride, padding
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with optional bias");
}