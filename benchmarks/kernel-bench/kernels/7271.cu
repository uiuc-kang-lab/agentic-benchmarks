#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 16
#define KERNEL_SIZE 3
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float load_input(
    const float* __restrict__ input,
    int b, int ic, int h, int w,
    int in_height, int in_width,
    int in_channels) {
    if (h >= 0 && h < in_height && w >= 0 && w < in_width)
        return input[((b * gridDim.z + ic) * in_height + h) * in_width + w];
    return 0.0f;
}

__device__ __forceinline__ float get_weight(
    const float* __restrict__ weight,
    int oc, int ic, int kh, int kw,
    int in_channels) {
    return weight[((oc * in_channels + ic) * KERNEL_SIZE + kh) * KERNEL_SIZE + kw];
}

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
    
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int oc = blockIdx.z % out_channels;
    const int b = blockIdx.z / out_channels;

    if (ty >= out_height || tx >= out_width || b >= batch_size) return;

    float sum = bias ? bias[oc] : 0.0f;
    const int in_start_h = ty * stride - padding;
    const int in_start_w = tx * stride - padding;

    #pragma unroll
    for (int ic = 0; ic < in_channels; ++ic) {
        #pragma unroll
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            const int h = in_start_h + kh * dilation;
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                const int w = in_start_w + kw * dilation;
                sum += load_input(input, b, ic, h, w, in_height, in_width) *
                       get_weight(weight, oc, ic, kh, kw, in_channels);
            }
        }
    }

    output[((b * out_channels + oc) * out_height + ty) * out_width + tx] = sum;
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
    if (bias.has_value()) CHECK_INPUT(bias.value());

    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto in_height = x.size(2);
    const auto in_width = x.size(3);
    const auto out_channels = weight.size(0);

    const auto out_height = (in_height + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) / stride + 1;
    const auto out_width = (in_width + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    const dim3 blocks(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );
    const dim3 threads(TILE_SIZE, TILE_SIZE);

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
    m.def("forward", &forward, "Optimized CUDA conv2d with modular functions");
}