#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void optimized_depthwise_conv2d_kernel(
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
    int c_out = blockIdx.x * blockDim.y + threadIdx.y;
    int spatial_idx = blockIdx.y * blockDim.x + threadIdx.x;
    int w_out = spatial_idx % out_w;
    int h_out = spatial_idx / out_w;
    int b = blockIdx.z;

    if (b >= batch_size || c_out >= out_channels || h_out >= out_h || w_out >= out_w) return;

    int g = c_out / channels_per_group;
    int m = c_out % channels_per_group;

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h_in = h_out * stride_h - padding_h + kh * dilation_h;
            int w_in = w_out * stride_w - padding_w + kw * dilation_w;
            
            if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                int input_idx = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
                int weight_idx = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw;
                sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
            }
        }
    }

    if (bias) sum += bias[c_out];
    
    output[((b * out_channels + c_out) * out_h + h_out) * out_w + w_out] = sum;
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
    TORCH_CHECK(x.device().is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be CUDA tensor");
    if (bias) TORCH_CHECK(bias->device().is_cuda(), "bias must be CUDA tensor");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = groups * weight.size(1);
    int channels_per_group = out_channels / groups;

    int out_h = (in_h + 2*padding_h - dilation_h*(kernel_h-1) - 1)/stride_h + 1;
    int out_w = (in_w + 2*padding_w - dilation_w*(kernel_w-1) - 1)/stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    dim3 threads(32, 4);  // 128 threads per block
    dim3 channels_blocks((out_channels + threads.y-1)/threads.y, 
                        (out_h*out_w + threads.x-1)/threads.x,
                        batch_size);

    optimized_depthwise_conv2d_kernel<<<channels_blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
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
    m.def("forward", &forward, "Optimized Depthwise Conv2D");
}