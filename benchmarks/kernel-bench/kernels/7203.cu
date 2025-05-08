#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int padding, int dilation,
    int groups) {

    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int ow = blockIdx.z * blockDim.z + threadIdx.z;
    const int oc = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.w;

    if (oc >= out_channels || oh >= out_height || ow >= out_width || n >= (input ? input[0] : 0)) return;

    const int group_size = out_channels / groups;
    const int g = oc / group_size;
    const int in_channels_per_group = in_channels / groups;

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels_per_group; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int ih = oh * stride - padding + kh * dilation;
                const int iw = ow * stride - padding + kw * dilation;
                
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    const int input_idx = ((n * in_channels + g * in_channels_per_group + ic) * in_height + ih) * in_width + iw;
                    const int weight_idx = ((oc * in_channels_per_group + ic) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (bias) sum += bias[oc];

    const int output_idx = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
    output[output_idx] = sum;
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

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    const int threads_x = 4;
    const int threads_y = 16;
    const int threads_z = 16;

    const dim3 threads(threads_x, threads_y, threads_z);
    const dim3 blocks(
        (out_channels + threads_x - 1) / threads_x,
        (out_height + threads_y - 1) / threads_y,
        (out_width + threads_z - 1) / threads_z,
        batch_size);

    conv2d_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        in_channels, in_height, in_width,
        out_channels, out_height, out_width,
        kernel_size, stride, padding, dilation,
        groups);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA forward function for 2D convolution");
}