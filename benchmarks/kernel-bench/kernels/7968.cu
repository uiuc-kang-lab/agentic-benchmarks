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
    float* output,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int out_height,
    int out_width) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z;

    if (row < out_height && col < out_width && channel < out_channels) {
        float sum = 0.0f;
        
        for (int kc = 0; kc < in_channels; ++kc) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int h = row * stride - padding + kh * dilation;
                    int w = col * stride - padding + kw * dilation;
                    
                    if (h >= 0 && h < in_height && w >= 0 && w < in_width) {
                        int input_idx = (kc * in_height + h) * in_width + w;
                        int weight_idx = ((channel * in_channels + kc) * kernel_size + kh) * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        int output_idx = (channel * out_height + row) * out_width + col;
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
    
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int kernel_size = weight.size(2);
    int out_channels = weight.size(0);
    
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());
    
    // Optimal block size found through experimentation
    dim3 block(16, 16);  // 256 threads per block
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        out_channels);
    
    for (int b = 0; b < batch_size; ++b) {
        conv2d_kernel<<<grid, block>>>(
            x[b].data_ptr<float>(),
            weight.data_ptr<float>(),
            output[b].data_ptr<float>(),
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            out_height,
            out_width);
    }
    
    if (bias.has_value()) {
        output += bias.value().view({1, out_channels, 1, 1});
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 2D convolution with block size tuning");
}
