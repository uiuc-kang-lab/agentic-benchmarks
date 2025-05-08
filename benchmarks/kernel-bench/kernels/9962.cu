#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int channels_per_group
) {
    const int total_outputs = batch_size * out_channels * out_h * out_w;
    const int tid = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int index = tid + i * blockDim.x;
        if (index >= total_outputs) return;

        const int w_out = index % out_w;
        index /= out_w;
        const int h_out = index % out_h;
        index /= out_h;
        const int c_out = index % out_channels;
        const int b = index / out_channels;

        const int g = c_out / channels_per_group;
        const int m = c_out % channels_per_group;

        float sum = 0.0f;
        #pragma unroll
        for (int kh = 0; kh < kernel_h; ++kh) {
            const int h_in = h_out * stride_h - padding_h + kh * dilation_h;
            #pragma unroll
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int w_in = w_out * stride_w - padding_w + kw * dilation_w;
                
                if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                    const int input_idx = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
                    const int weight_idx = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[c_out];
        }
        
        output[((b * out_channels + c_out) * out_h + h_out) * out_w + w_out] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
    }

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels = groups * weight.size(1);
    const int channels_per_group = out_channels / groups;
    const int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());
    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    const int total_outputs = batch_size * out_channels * out_h * out_w;
    const int blocks = (total_outputs + BLOCK_SIZE * ELEMENTS_PER_THREAD - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD);

    depthwise_conv2d_kernel<<<blocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise Conv2D Balanced Load forward (CUDA)");
}