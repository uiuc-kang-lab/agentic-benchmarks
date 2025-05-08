#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv2d_optimized_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int height_out,
    int width_out,
    int stride,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * height_out * width_out) return;

    int w_out = idx % width_out;
    idx /= width_out;
    int h_out = idx % height_out;
    idx /= height_out;
    int oc = idx % out_channels;
    int b = idx / out_channels;

    float sum = bias ? bias[oc] : 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            const int h_in_base = h_out * stride - pad_h + kh * dilation_h;
            if (h_in_base < 0 || h_in_base >= input_height) continue;
            
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int w_in = w_out * stride - pad_w + kw * dilation_w;
                if (w_in < 0 || w_in >= input_width) continue;

                int x_idx = b * in_channels * input_height * input_width
                           + ic * input_height * input_width
                           + h_in_base * input_width
                           + w_in;
                
                int w_idx = oc * in_channels * kernel_h * kernel_w
                           + ic * kernel_h * kernel_w
                           + kh * kernel_w
                           + kw;
                           
                sum += x[x_idx] * weight[w_idx];
            }
        }
    }

    output[b * out_channels * height_out * width_out +
           oc * height_out * width_out +
           h_out * width_out +
           w_out] = sum;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    std::tuple<int, int> padding,
    std::tuple<int, int> dilation) {

    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_height = x.size(2);
    const int input_width = x.size(3);

    const int out_channels = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int pad_h = std::get<0>(padding);
    const int pad_w = std::get<1>(padding);
    const int dilation_h = std::get<0>(dilation);
    const int dilation_w = std::get<1>(dilation);

    const int height_out = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1)/stride + 1;
    const int width_out = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1)/stride + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, x.options());
    
    const int total = batch_size * out_channels * height_out * width_out;
    if (total == 0) return output;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    conv2d_optimized_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_height,
        input_width,
        out_channels,
        kernel_h,
        kernel_w,
        height_out,
        width_out,
        stride,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Conv2D forward");
}