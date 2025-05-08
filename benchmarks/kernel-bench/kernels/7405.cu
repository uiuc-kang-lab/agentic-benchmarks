#include <torch/extension.h>

__global__ void conv_transpose2d_kernel(const float *x, const float *weight, float *output,
                                        int input_height, int input_width,
                                        int kernel_size, int stride, int padding,
                                        int output_height, int output_width) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < output_width && out_y < output_height) {
        float sum = 0.0f;

        for (int dy = 0; dy < kernel_size; ++dy) {
            for (int dx = 0; dx < kernel_size; ++dx) {
                int in_x = (out_x - dx) + padding;
                int in_y = (out_y - dy) + padding;

                if (in_x % stride == 0 && in_y % stride == 0) {
                    in_x /= stride;
                    in_y /= stride;
                    
                    if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height) {
                        // Index calculations
                        int input_index = in_y * input_width + in_x;
                        int weight_index = dy * kernel_size + dx;

                        sum += x[input_index] * weight[weight_index];
                    }
                }
            }
        }
        int output_index = out_y * output_width + out_x;
        output[output_index] = sum;
    }
}

// Forward function definition
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

    // Calculate output dimensions
    int batch_size = x.size(0);
    int input_height = x.size(2);
    int input_width = x.size(3);
    int kernel_size = weight.size(2);
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, weight.size(1), output_height, output_width}, x.options());

    // Launch kernel with optimized grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    int blocksX = (output_width + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int blocksY = (output_height + threadsPerBlock.y - 1) / threadsPerBlock.y;
    dim3 numBlocks(blocksX, blocksY);

    for (int b = 0; b < batch_size; ++b) {
        conv_transpose2d_kernel<<<numBlocks, threadsPerBlock>>>(
            x[b].data_ptr<float>(),
            weight.data_ptr<float>(),
            output[b].data_ptr<float>(),
            input_height,
            input_width,
            kernel_size,
            stride,
            padding,
            output_height,
            output_width);
    }

    if (bias.has_value()) {
        output += bias.value().view({1, -1, 1, 1});
    }

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2D forward with optimized grid (CUDA)");
}