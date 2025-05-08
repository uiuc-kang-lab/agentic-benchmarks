#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_kernel(const float* __restrict__ input, const float* __restrict__ weight, float* output, 
                              int input_height, int input_width, int kernel_size, int stride, int padding, int output_height, int output_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < output_width && idy < output_height) {
        float result = 0.0f;
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int x = idx * stride + i - padding;
                int y = idy * stride + j - padding;
                if (x >= 0 && x < input_width && y >= 0 && y < input_height) {
                    result += __ldg(&input[y * input_width + x]) * __ldg(&weight[i * kernel_size + j]);
                }
            }
        }
        output[idy * output_width + idx] = result;
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

    int batch_size = x.size(0);
    int input_channels = x.size(1);
    int input_height = x.size(2);
    int input_width = x.size(3);
    int kernel_size = weight.size(2);

    int output_height = (input_height - kernel_size + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_size + 2 * padding) / stride + 1;

    torch::Tensor output = torch::empty({batch_size, input_channels, output_height, output_width}, x.options());

    const int threads = 32;
    const dim3 blocks((output_width + threads - 1) / threads, (output_height + threads - 1) / threads);
    const dim3 threads_per_block(threads, threads);

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < input_channels; ++c) {
            conv2d_kernel<<<blocks, threads_per_block>>>(
                x.data_ptr<float>() + b * input_channels * input_height * input_width + c * input_height * input_width,
                weight.data_ptr<float>() + c * kernel_size * kernel_size * input_channels,
                output.data_ptr<float>() + b * input_channels * output_height * output_width + c * output_height * output_width,
                input_height,
                input_width,
                kernel_size,
                stride,
                padding,
                output_height,
                output_width);
        }
    }

    return bias.has_value() ? output + bias.value().view({1, -1, 1, 1}) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for optimized 2D convolution with optional bias");
}