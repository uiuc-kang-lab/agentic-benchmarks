#include <torch/extension.h>

// Forward function definition
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int output_height,
    int output_width) {

    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_col < output_width && out_row < output_height && out_channel < output_channels) {
        for (int b = 0; b < batch_size; ++b) {
            float sum = 0.0f;
            for (int c = 0; c < input_channels; ++c) {
                for (int k_row = 0; k_row < kernel_size; ++k_row) {
                    for (int k_col = 0; k_col < kernel_size; ++k_col) {
                        int in_row = out_row - k_row + padding;
                        int in_col = out_col - k_col + padding;
                        if (in_row % stride == 0 && in_col % stride == 0) {
                            in_row /= stride;
                            in_col /= stride;
                            if (in_row >= 0 && in_row < input_height && in_col >= 0 && in_col < input_width) {
                                sum += weight[out_channel * input_channels * kernel_size * kernel_size + c * kernel_size * kernel_size + k_row * kernel_size + k_col] *
                                       x[b * input_channels * input_height * input_width + c * input_height * input_width + in_row * input_width + in_col];
                            }
                        }
                    }
                }
            }
            output[b * output_channels * output_height * output_width + out_channel * output_height * output_width + out_row * output_width + out_col] = sum;
        }
    }
}

// Wrapper function
torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    // Ensure inputs are on CUDA and contiguous
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
    }

    auto batch_size = x.size(0);
    auto input_channels = x.size(1);
    auto input_height = x.size(2);
    auto input_width = x.size(3);
    auto output_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    auto output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    auto output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, output_channels, output_height, output_width}, x.options());

    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((output_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (output_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   output_channels);

    conv_transpose2d_kernel<<<numBlocks, threadsPerBlock>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        output_height,
        output_width);

    if (bias.has_value()) {
        output += bias.value().view({1, output_channels, 1, 1});
    }

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA)");
}