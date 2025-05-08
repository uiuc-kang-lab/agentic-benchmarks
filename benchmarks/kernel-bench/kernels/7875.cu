#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Declare constant memory for weights (limited to 64KB on most CUDA devices)
__constant__ float const_weight[16384]; // 64KB / 4 bytes = 16384 float elements

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding) {

    const int out_h = (height + 2 * padding - kernel_h) / stride + 1;
    const int out_w = (width + 2 * padding - kernel_w) / stride + 1;
    const int total = batch_size * out_channels * out_h * out_w;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= total) return;

    const int b = index / (out_channels * out_h * out_w);
    const int rem = index % (out_channels * out_h * out_w);
    const int oc = rem / (out_h * out_w);
    const int rem2 = rem % (out_h * out_w);
    const int h = rem2 / out_w;
    const int w = rem2 % out_w;

    float sum = 0.0f;

    const int h_start = max(0, h * stride - padding);
    const int h_end = min(kernel_h, height - h * stride + padding);
    const int w_start = max(0, -w * stride + padding);
    const int w_end = min(kernel_w, width - w * stride + padding);

    #pragma unroll 4
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = h_start; kh < h_end; ++kh) {
            const int h_in = h * stride + kh - padding;
            #pragma unroll 4
            for (int kw = w_start; kw < w_end; ++kw) {
                const int w_in = w * stride + kw - padding;
                const float input_val = __ldg(&input[((b * in_channels + ic) * height + h_in) * width + w_in]);
                const int weight_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                sum += input_val * const_weight[weight_idx];
            }
        }
    }

    output[((b * out_channels + oc) * out_h + h) * out_w + w] = sum;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    
    if (dilation != 1 || groups != 1) {
        return torch::conv2d(x, weight, bias,
                           {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
    }

    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    // Check if weights fit in constant memory
    TORCH_CHECK(weight.numel() <= 16384, "Weight tensor too large for constant memory");
    
    // Copy weights to constant memory
    cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), 
                       weight.numel() * sizeof(float));
    
    const int out_h = (height + 2 * padding - kernel_h) / stride + 1;
    const int out_w = (width + 2 * padding - kernel_w) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());

    constexpr int BLOCK_SIZE = 256;
    const int total = batch_size * out_channels * out_h * out_w;
    const int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    conv2d_kernel<<<blocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
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