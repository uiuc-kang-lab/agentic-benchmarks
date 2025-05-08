#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

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
    int output_padding,
    int groups) {
    
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (oh >= out_height || ow >= out_width) return;
    
    int g = blockIdx.z / (out_channels / groups);
    int oc = blockIdx.z % (out_channels / groups);
    
    float val = 0.0f;
    
    for (int ic = 0; ic < in_channels/groups; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = (oh - kh + padding) / stride;
                int iw = (ow - kw + padding) / stride;
                
                if ((oh - kh + padding) %% stride != 0) continue;
                if ((ow - kw + padding) %% stride != 0) continue;
                
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    int input_idx = ((g * (in_channels/groups) + ic) * in_height + ih) * in_width + iw;
                    int weight_idx = ((g * (out_channels/groups) + oc) * (in_channels/groups) + ic) * kernel_size * kernel_size + kh * kernel_size + kw;
                    val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    if (bias) {
        val += bias[g * (out_channels/groups) + oc];
    }
    
    int output_idx = ((g * (out_channels/groups) + oc) * out_height + oh) * out_width + ow;
    output[output_idx] = val;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride,
    py::object padding,
    py::object output_padding,
    int64_t groups) {
    
    auto in_sizes = x.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = in_sizes[0];
    int in_channels = in_sizes[1];
    int in_height = in_sizes[2];
    int in_width = in_sizes[3];
    
    int out_channels = weight_sizes[1] * groups;
    int kernel_size = weight_sizes[2];
    
    auto stride_vec = parseIntArrayRef(stride);
    auto padding_vec = parseIntArrayRef(padding);
    auto output_padding_vec = parseIntArrayRef(output_padding);
    
    int stride_h = stride_vec[0];
    int padding_h = padding_vec[0];
    int output_padding_h = output_padding_vec[0];
    
    int out_height = (in_height - 1) * stride_h - 2 * padding_h + kernel_size + output_padding_h;
    int out_width = (in_width - 1) * stride_vec[1] - 2 * padding_vec[1] + kernel_size + output_padding_vec[1];
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());
    
    dim3 blocks(
        (out_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (out_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
        groups * (out_channels / groups)
    );
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    
    for (int b = 0; b < batch_size; ++b) {
        auto x_accessor = x.accessor<float,4>()[b];
        auto weight_accessor = weight.accessor<float,4>();
        auto output_accessor = output.accessor<float,4>()[b];
        
        conv_transpose2d_kernel<<<blocks, threads>>>(
            x_accessor.data(),
            weight_accessor.data(),
            bias ? bias->data_ptr<float>() : nullptr,
            output_accessor.data(),
            in_channels,
            out_channels,
            in_height,
            in_width,
            out_height,
            out_width,
            kernel_size,
            stride_h,
            padding_h,
            output_padding_h,
            groups);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d forward optimized",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}