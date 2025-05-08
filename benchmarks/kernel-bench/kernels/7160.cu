#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_kernel(float* input, float* weight, float* output, int batch_size, int in_channels, int out_channels,
                              int input_h, int input_w, int kernel_h, int kernel_w, int stride, int padding, int dilation) {
    int out_w = (input_w + 2*padding - dilation*(kernel_w - 1) - 1) / stride + 1;
    int out_h = (input_h + 2*padding - dilation*(kernel_h - 1) - 1) / stride + 1;
    int out_channel = blockIdx.x;
    int batch = blockIdx.y;
    int x = threadIdx.x;
    int y = threadIdx.y;

    float sum = 0.0f;
    for (int c = 0; c < in_channels; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = y * stride - padding + kh * dilation;
                int iw = x * stride - padding + kw * dilation;
                if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                    int input_idx = ((batch * in_channels + c) * input_h + ih) * input_w + iw;
                    int weight_idx = ((out_channel * in_channels + c) * kernel_h + kh) * kernel_w + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    int output_idx = ((batch * out_channels + out_channel) * out_h + y) * out_w + x;
    atomicAdd(&output[output_idx], sum);
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

    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto input_h = x.size(2);
    const auto input_w = x.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);

    auto opts = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::zeros({batch_size, out_channels, (input_h + 2*padding - dilation*(kernel_h - 1) - 1)/stride + 1, (input_w + 2*padding - dilation*(kernel_w - 1) - 1)/stride + 1}, opts);

    dim3 threads(16, 16);
    dim3 blocks(out_channels, batch_size);

    conv2d_kernel<<<blocks, threads>>>(x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, input_h, input_w, kernel_h, kernel_w, stride, padding, dilation);

    if (bias.has_value()) {
        output += bias.value().view({1, out_channels, 1, 1});
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA forward function for 2D convolution with atomic operations");
}