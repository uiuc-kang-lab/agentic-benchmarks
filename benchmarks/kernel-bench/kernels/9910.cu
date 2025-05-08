#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_h,
    int input_w,
    int out_channels,
    int output_h,
    int output_w,
    int kernel_size,
    int stride,
    int padding,
    int channels_per_group
) {
    int total_elements = batch_size * out_channels * output_h * output_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Compute output indices (w_out, h_out, oc, b) from linear index
    int linear_idx = idx;
    int w_out = linear_idx % output_w;
    linear_idx /= output_w;
    int h_out = linear_idx % output_h;
    linear_idx /= output_h;
    int oc = linear_idx % out_channels;
    linear_idx /= out_channels;
    int b = linear_idx;  

    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    float sum = 0.0f;

    // Pre-compute base indices for input and weight
    int input_batch_offset = b * (in_channels * input_h * input_w);
    int input_channel_offset = in_ch * (input_h * input_w);
    int weight_offset = in_ch * (channels_per_group * kernel_size * kernel_size)
                      + weight_ch * (kernel_size * kernel_size);

    #pragma unroll
    for (int kh = 0; kh < kernel_size; ++kh) {
        int h_in = h_out * stride + kh - padding;
        if (h_in >= 0 && h_in < input_h) {
            #pragma unroll
            for (int kw = 0; kw < kernel_size; ++kw) {
                int w_in = w_out * stride + kw - padding;
                if (w_in >= 0 && w_in < input_w) {
                    int input_idx = input_batch_offset + input_channel_offset + h_in * input_w + w_in;
                    int weight_idx = weight_offset + kh * kernel_size + kw;
                    sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += __ldg(&bias[oc]);
    }

    output[b * out_channels * output_h * output_w +
           oc * output_h * output_w +
           h_out * output_w +
           w_out] = sum;
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int block_size = BLOCK_SIZE
) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input and weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_contiguous(), "Bias must be contiguous");
    }
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int kernel_size = weight.size(2);
    int channels_per_group = weight.size(1);
    int out_channels = in_channels * channels_per_group;

    if (bias.has_value()) {
        TORCH_CHECK(bias->size(0) == out_channels, "Bias size mismatch");
    }

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    int total_elements = batch_size * out_channels * output_h * output_w;
    int threads = block_size;
    int blocks = (total_elements + threads - 1) / threads;

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_h,
        input_w,
        out_channels,
        output_h,
        output_w,
        kernel_size,
        stride,
        padding,
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Depthwise 2D Convolution (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("block_size") = BLOCK_SIZE);
}
