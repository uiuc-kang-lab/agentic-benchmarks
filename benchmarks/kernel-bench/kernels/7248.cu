#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_kernel(float* x, float* weight, float* bias, float* output, int stride, int padding, int dilation, int groups, int in_channels, int out_channels, int kernel_size, int H, int W, int out_H, int out_W) {
    int n = blockIdx.x;
    int oc = blockIdx.y;
    int oh = threadIdx.y;
    int ow = threadIdx.x;

    float value = 0.0f;

    #pragma unroll
    for (int ic = 0; ic < in_channels; ++ic) {
        #pragma unroll
        for (int kh = 0; kh < kernel_size; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh * stride - padding + kh * dilation;
                int iw = ow * stride - padding + kw * dilation;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    value += x[n * in_channels * H * W + ic * H * W + ih * W + iw] * 
                             weight[oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw];
                }
            }
        }
    }

    if (bias != nullptr) {
        value += bias[oc];
    }

    output[n * out_channels * out_H * out_W + oc * out_H * out_W + oh * out_W + ow] = value;
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

    auto output = torch::empty({x.size(0), weight.size(0), (x.size(2) + 2 * padding - dilation * (weight.size(2) - 1) - 1) / stride + 1, (x.size(3) + 2 * padding - dilation * (weight.size(3) - 1) - 1) / stride + 1}, x.options());

    int in_channels = x.size(1);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int H = x.size(2);
    int W = x.size(3);
    int out_H = output.size(2);
    int out_W = output.size(3);

    dim3 threads(out_W, out_H);
    dim3 blocks(x.size(0), out_channels);

    conv2d_kernel<<<blocks, threads>>>(x.data_ptr<float>(), weight.data_ptr<float>(), bias.has_value() ? bias.value().data_ptr<float>() : nullptr, output.data_ptr<float>(), stride, padding, dilation, groups, in_channels, out_channels, kernel_size, H, W, out_H, out_W);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution");
}