#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define constant memory for weights and bias
__constant__ float const_weight[1024];
__constant__ float const_bias[64];

__global__ void conv2d_kernel(const float* input, float* output, int input_size, int kernel_size, int stride, int padding, int output_size) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_x < output_size && out_y < output_size) {
        float value = 0.0f;
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int in_x = out_x * stride + i - padding;
                int in_y = out_y * stride + j - padding;
                if (in_x >= 0 && in_x < input_size && in_y >= 0 && in_y < input_size) {
                    value += input[in_y * input_size + in_x] * const_weight[i * kernel_size + j];
                }
            }
        }
        if (blockIdx.z < 64) { // Assuming 64 output channels
            value += const_bias[blockIdx.z];
        }
        output[blockIdx.z * output_size * output_size + out_y * output_size + out_x] = value;
    }
}

void load_constants(torch::Tensor weight, torch::optional<torch::Tensor> bias) {
    cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), weight.numel() * sizeof(float));
    if (bias.has_value()) {
        cudaMemcpyToSymbol(const_bias, bias.value().data_ptr<float>(), bias.value().numel() * sizeof(float));
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    int input_size = x.size(2); // Assuming square input
    int kernel_size = weight.size(2); // Assuming square kernel
    int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;

    auto output = torch::empty({x.size(0), weight.size(0), output_size, output_size}, x.options());

    load_constants(weight, bias);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (output_size + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   weight.size(0));

    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(x.data_ptr<float>(), output.data_ptr<float>(), input_size, kernel_size, stride, padding, output_size);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with constant memory optimization");
}