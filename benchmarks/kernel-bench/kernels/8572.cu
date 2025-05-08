#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256  // Start with 256, experiment with 32/64/128/512

template<int BLOCK>
__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding) {
    
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    int spatial_size = out_height * out_width;
    int c_stride = spatial_size * out_channels;

    if (idx < c_stride) {
        int oc = idx / spatial_size;
        int oh = (idx % spatial_size) / out_width;
        int ow = idx % out_width;
        
        float val = 0.0f;
        
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int ih = (oh + padding - kh) / stride;
                    int iw = (ow + padding - kw) / stride;
                    
                    if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width && 
                        (oh + padding - kh) % stride == 0) {
                        int input_idx = ic * in_height * in_width + ih * in_width + iw;
                        int weight_idx = oc * in_channels * kernel_size * kernel_size + 
                                       ic * kernel_size * kernel_size + kh * kernel_size + kw;
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        if (bias) val += bias[oc];
        output[idx] = val;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int output_padding,
    int groups) {
    
    auto input = x.contiguous();
    auto w = weight.contiguous();
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int kernel_size = weight.size(2);
    int out_channels = weight.size(1);
    
    int out_height = (in_height - 1) * stride + kernel_size - 2 * padding + output_padding;
    int out_width = (in_width - 1) * stride + kernel_size - 2 * padding + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());
    
    int num_blocks = (out_channels * out_height * out_width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int b = 0; b < batch_size; ++b) {
        conv_transpose2d_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(
            input.data_ptr<float>() + b * in_channels * in_height * in_width,
            w.data_ptr<float>(),
            bias ? bias->data_ptr<float>() : nullptr,
            output.data_ptr<float>() + b * out_channels * out_height * out_width,
            in_channels,
            out_channels,
            in_height,
            in_width,
            out_height,
            out_width,
            kernel_size,
            stride,
            padding,
            output_padding);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d optimized",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}