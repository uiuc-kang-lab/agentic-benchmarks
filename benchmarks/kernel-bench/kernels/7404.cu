#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel function for transposed convolution
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int input_size,
    int kernel_size,
    int output_size,
    int stride,
    int padding,
    int output_padding,
    int groups) {

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < output_size && out_y < output_size) {
        float value = 0.0f;
        for (int c = 0; c < groups; ++c) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                for (int ky = 0; ky < kernel_size; ++ky) {
                    int in_x = out_x - kx + padding;
                    int in_y = out_y - ky + padding;
                    if (in_x % stride == 0 && in_y % stride == 0) {
                        in_x /= stride;
                        in_y /= stride;
                        if (in_x >= 0 && in_x < input_size && in_y >= 0 && in_y < input_size) {
                            value += x[(c * input_size + in_y) * input_size + in_x] * 
                                     weight[(c * kernel_size + ky) * kernel_size + kx];
                        }
                    }
                }
            }
        }
        atomicAdd(&output[out_y * output_size + out_x], value);
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

    int input_size = x.size(2);
    int kernel_size = weight.size(2);
    int output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({x.size(0), weight.size(0), output_size, output_size}, x.options());

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (output_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    conv_transpose2d_kernel<<<numBlocks, threadsPerBlock>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        input_size,
        kernel_size,
        output_size,
        stride,
        padding,
        output_padding,
        groups);

    if (bias.has_value()) {
        output += bias.value().view({1, -1, 1, 1});
    }

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA)");
}