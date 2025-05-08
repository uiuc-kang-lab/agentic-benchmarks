#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

__device__ float compute_transpose_convolution(float *input, float *weight, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int h, int w) {
    float sum = 0.0f;
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int ih = h - kh + pad_h;
            int iw = w - kw + pad_w;
            if (ih % stride_h == 0 && iw % stride_w == 0) {
                ih /= stride_h;
                iw /= stride_w;
                sum += input[ih * gridDim.x + iw] * weight[kh * kernel_w + kw];
            }
        }
    }
    return sum;
}

__global__ void conv_transpose2d_kernel(float *input, float *weight, float *output, int input_h, int input_w, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int output_h, int output_w) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (h < output_h && w < output_w) {
        output[h * output_w + w] = compute_transpose_convolution(input, weight, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, h, w);
    }
}

void conv_transpose2d_forward(  
    torch::Tensor x,  
    torch::Tensor weight,  
    torch::Tensor output,  
    std::vector<int64_t> stride,  
    std::vector<int64_t> padding  
) {
    const auto input_h = x.size(0);
    const auto input_w = x.size(1);
    const auto kernel_h = weight.size(0);
    const auto kernel_w = weight.size(1);

    const dim3 blockDim(16, 16);
    const dim3 gridDim((output.size(1) + blockDim.x - 1) / blockDim.x, (output.size(0) + blockDim.y - 1) / blockDim.y);

    conv_transpose2d_kernel<<<gridDim, blockDim>>>(x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), input_h, input_w, kernel_h, kernel_w, stride[0], stride[1], padding[0], padding[1], output.size(0), output.size(1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward optimized",
          py::arg("x"),
          py::arg("weight"),
          py::arg("output"),
          py::arg("stride"),
          py::arg("padding"));
}