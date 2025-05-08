#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

namespace py = pybind11;

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
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
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    const int total_elements = batch_size * out_channels * out_height * out_width;
    
    for (int idx = tid; idx < total_elements; idx += stride) {
        const int w = idx % out_width;
        const int h = (idx / out_width) % out_height;
        const int c = (idx / (out_width * out_height)) % out_channels;
        const int b = idx / (out_width * out_height * out_channels);
        
        float sum = 0.0f;
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    const int in_h = (h + pad_h - kh) / stride_h;
                    const int in_w = (w + pad_w - kw) / stride_w;
                    
                    if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                        const int input_idx = ((b * in_channels + ic) * in_height + in_h) * in_width + in_w;
                        const int weight_idx = ((c * in_channels + ic) * kernel_height + kh) * kernel_width + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto in_height = x.size(2);
    const auto in_width = x.size(3);
    const auto out_channels = weight.size(1);
    const auto kernel_height = weight.size(2);
    const auto kernel_width = weight.size(3);
    
    const auto out_height = (in_height - 1) * stride[0] - 2 * padding[0] + kernel_height;
    const auto out_width = (in_width - 1) * stride[1] - 2 * padding[1] + kernel_width;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width},
                              x.options());
    
    const int threads = 256;
    const int blocks = (batch_size * out_channels * out_height * out_width + threads - 1) / threads;
    
    conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_height,
        kernel_width,
        out_height,
        out_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1]
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}