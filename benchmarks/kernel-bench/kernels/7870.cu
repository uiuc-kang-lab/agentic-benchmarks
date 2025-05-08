#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void calculate_output_coords(
    int index, int out_channels, int out_h, int out_w,
    int& b, int& oc, int& h, int& w
) {
    b = index / (out_channels * out_h * out_w);
    int rem = index % (out_channels * out_h * out_w);
    oc = rem / (out_h * out_w);
    rem = rem % (out_h * out_w);
    h = rem / out_w;
    w = rem % out_w;
}

__device__ void calculate_kernel_bounds(
    int h, int w, int stride, int padding,
    int height, int width, int kernel_h, int kernel_w,
    int& h_start, int& h_end, int& w_start, int& w_end
) {
    h_start = max(0, -h * stride + padding);
    h_end = min(kernel_h, height - h * stride + padding);
    w_start = max(0, -w * stride + padding);
    w_end = min(kernel_w, width - w * stride + padding);
}

__device__ float convolution_sum(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int b, int oc, int h, int w,
    int in_channels, int height, int width,
    int kernel_h, int kernel_w,
    int stride, int padding,
    int h_start, int h_end, int w_start, int w_end
) {
    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = h_start; kh < h_end; ++kh) {
            const int h_in = h * stride + kh - padding;
            #pragma unroll 4
            for (int kw = w_start; kw < w_end; ++kw) {
                const int w_in = w * stride + kw - padding;
                const float input_val = __ldg(&input[((b * in_channels + ic) * height + h_in) * width + w_in]);
                const float weight_val = __ldg(&weight[((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw]);
                sum += input_val * weight_val;
            }
        }
    }
    return sum;
}

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding
) {
    const int out_h = (height + 2 * padding - kernel_h) / stride + 1;
    const int out_w = (width + 2 * padding - kernel_w) / stride + 1;
    const int total = batch_size * out_channels * out_h * out_w;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= total) return;

    int b, oc, h, w;
    calculate_output_coords(index, out_channels, out_h, out_w, b, oc, h, w);

    int h_start, h_end, w_start, w_end;
    calculate_kernel_bounds(h, w, stride, padding, height, width,
                           kernel_h, kernel_w, h_start, h_end, w_start, w_end);

    const float sum = convolution_sum(input, weight, b, oc, h, w,
                                     in_channels, height, width,
                                     kernel_h, kernel_w, stride, padding,
                                     h_start, h_end, w_start, w_end);

    output[((b * out_channels + oc) * out_h + h) * out_w + w] = sum;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x);
    CHECK_CUDA(weight); CHECK_CONTIGUOUS(weight);

    if (dilation != 1 || groups != 1) {
        return torch::conv2d(x, weight, bias, {stride, stride},
                           {padding, padding}, {dilation, dilation}, groups);
    }

    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto height = x.size(2);
    const auto width = x.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);

    const int out_h = (height + 2 * padding - kernel_h) / stride + 1;
    const int out_w = (width + 2 * padding - kernel_w) / stride + 1;
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());

    constexpr int BLOCK_SIZE = 256;
    const int total = batch_size * out_channels * out_h * out_w;
    const int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    conv2d_kernel<<<blocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_h,
        kernel_w,
        stride,
        padding
    );

    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution");
}