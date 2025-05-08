#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Modular device function to compute a single output pixel
__device__ inline float compute_conv_transposed_element(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int n,
    int oc,
    int out_y,
    int out_x,
    int in_channels,
    int in_h,
    int in_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int out_channels
) {
    float sum = 0.0f;
    int input_channel_stride = in_h * in_w;
    int base_y = out_y + pad_h;
    int base_x = out_x + pad_w;
    
    for (int ky = 0; ky < kernel_h; ky++) {
        int t_y = base_y - ky;
        if (t_y % stride_h != 0) continue;
        int in_y = t_y / stride_h;
        if (in_y < 0 || in_y >= in_h) continue;
        
        for (int kx = 0; kx < kernel_w; kx++) {
            int t_x = base_x - kx;
            if (t_x % stride_w != 0) continue;
            int in_x = t_x / stride_w;
            if (in_x < 0 || in_x >= in_w) continue;

            int input_index_base = (n * in_channels) * input_channel_stride + in_y * in_w + in_x;
            int weight_offset = ky * kernel_w + kx;

            for (int ic = 0; ic < in_channels; ic++) {
                int input_idx = input_index_base + ic * input_channel_stride;
                int weight_idx = ic * (out_channels * kernel_h * kernel_w) + oc * (kernel_h * kernel_w) + weight_offset;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    return sum;
}

// Main kernel that computes one output element per thread
__global__ void conv_transpose2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int out_h,
    int out_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    bool has_bias
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_channels * out_h * out_w;
    if (index >= total) return;

    // Decode the flat index into (n, oc, out_y, out_x)
    int out_x = index % out_w;
    int tmp = index / out_w;
    int out_y = tmp % out_h;
    tmp = tmp / out_h;
    int oc = tmp % out_channels;
    int n = tmp / out_channels;

    float value = has_bias ? bias[oc] : 0.0f;
    value += compute_conv_transposed_element(input, weight, n, oc, out_y, out_x,
                                               in_channels, in_h, in_w,
                                               kernel_h, kernel_w,
                                               stride_h, stride_w,
                                               pad_h, pad_w, out_channels);
    output[index] = value;
}

// Host function to set up and launch the kernel
torch::Tensor conv_transpose2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    int N = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);

    int out_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int stride_h = stride[0];
    int stride_w = stride[1];
    int pad_h = padding[0];
    int pad_w = padding[1];

    // Compute output dimensions using the transposed convolution formula
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w;

    auto output = torch::zeros({N, out_channels, out_h, out_w}, input.options());
    
    int total = N * out_channels * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    bool has_bias = (bias.has_value() && bias.value().numel() > 0);
    const float* bias_ptr = has_bias ? bias.value().data_ptr<float>() : nullptr;

    conv_transpose2d_forward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, in_channels, in_h, in_w,
        out_channels, kernel_h, kernel_w,
        out_h, out_w,
        stride_h, stride_w,
        pad_h, pad_w,
        has_bias
    );
    
    return output;
}

// Entry point exposed to Python
torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    c10::optional<torch::Tensor> bias = c10::nullopt;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
    }
    return conv_transpose2d_forward_cuda(input, weight, bias, stride, padding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward (modular device functions)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}
