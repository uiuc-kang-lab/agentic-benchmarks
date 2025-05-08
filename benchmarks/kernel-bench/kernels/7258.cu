#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 32
#define KERNEL_SIZE 3
#define UNROLL_FACTOR 4

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation) {
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * TILE_SIZE;
    const int by = blockIdx.y * TILE_SIZE;
    const int b = blockIdx.z / out_channels;
    const int oc = blockIdx.z % out_channels;

    float sum = bias ? bias[oc] : 0.0f;

    #pragma unroll UNROLL_FACTOR
    for (int ic = 0; ic < in_channels; ic++) {
        #pragma unroll
        for (int kh = 0; kh < KERNEL_SIZE; kh++) {
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                const int h = (by + ty) * stride - padding + kh * dilation;
                const int w = (bx + tx) * stride - padding + kw * dilation;
                
                if (h >= 0 && h < in_height && w >= 0 && w < in_width) {
                    const int input_idx = ((b * in_height + h) * in_width + w) * in_channels + ic;
                    const int weight_idx = ((oc * KERNEL_SIZE + kh) * KERNEL_SIZE + kw) * in_channels + ic;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    const int out_h = by + ty;
    const int out_w = bx + tx;
    if (out_h < out_height && out_w < out_width) {
        output[((b * out_channels + oc) * out_height + out_h) * out_width + out_w] = sum;
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
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto in_height = x.size(2);
    const auto in_width = x.size(3);
    const auto out_channels = weight.size(0);
    
    const auto out_height = (in_height + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) / stride + 1;
    const auto out_width = (in_width + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );

    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        stride,
        padding,
        dilation
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with coalesced memory access");
}