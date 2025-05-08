#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_optimized_kernel(
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
    int oc = blockIdx.x * blockDim.y + threadIdx.y;
    int w_out = blockIdx.y * blockDim.x + threadIdx.x;
    int b = blockIdx.z;

    if (b >= batch_size || oc >= out_channels || w_out >= output_w) return;

    int h_out = blockIdx.z % output_h;
    b = blockIdx.z / output_h;

    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    float sum = 0.0f;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_in = h_out * stride + kh - padding;
            int w_in = w_out * stride + kw - padding;

            if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                int input_idx = b * (in_channels * input_h * input_w)
                              + in_ch * (input_h * input_w)
                              + h_in * input_w
                              + w_in;

                int weight_idx = in_ch * (channels_per_group * kernel_size * kernel_size)
                               + weight_ch * (kernel_size * kernel_size)
                               + kh * kernel_size
                               + kw;

                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
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
    int padding
) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input and weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_contiguous(), "Bias must be contiguous");
    }
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D tensor");

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

    dim3 threads(32, 8);  // x: output_w, y: channels
    dim3 blocks(
        (out_channels + threads.y - 1) / threads.y,
        (output_w + threads.x - 1) / threads.x,
        batch_size * output_h
    );

    const float* bias_ptr = bias ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_optimized_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution Optimized (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
