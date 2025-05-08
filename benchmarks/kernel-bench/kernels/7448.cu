#include <torch/extension.h>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int out_height,
    const int out_width) {

    // 2D thread indexing
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.z / out_channels;
    const int oc = blockIdx.z % out_channels;

    if (h < out_height && w < out_width && b < batch_size) {
        float sum = bias ? bias[oc] : 0.0f;

        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    // For transposed conv, we need to flip the kernel indices and adjust the input indexing
                    const int h_in = (h + padding - kh * stride) / stride;
                    const int w_in = (w + padding - kw * stride) / stride;

                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {

                        const int input_idx = ((b * in_channels + ic) * in_height + h_in) * in_width + w_in;
                        const int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;

                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        const int output_idx = ((b * out_channels + oc) * out_height + h) * out_width + w;
        output[output_idx] = sum;
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(1) * groups;
    const int kernel_size = weight.size(2);
    
    const int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width},
                              x.options());

    const dim3 threads(16, 16);
    const dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        batch_size * out_channels
    );

    conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        out_height,
        out_width
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA)");
}