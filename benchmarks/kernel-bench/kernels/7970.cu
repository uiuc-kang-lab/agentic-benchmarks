#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Device function for convolution operation
__device__ float conv2d_element(const float* input, const float* kernel, int kernel_size, int input_width, int input_height, int x, int y) {
    float result = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int input_x = x + i - kernel_size / 2;
            int input_y = y + j - kernel_size / 2;
            if (input_x < input_width && input_y < input_height) {
                result += input[input_y * input_width + input_x] * kernel[j * kernel_size + i];
            }
        }
    }
    return result;
}

// Kernel function
__global__ void conv2d_kernel(const float* input, const float* kernel, float* output, int kernel_size, int input_width, int input_height, int output_width, int output_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_width && y < output_height) {
        output[y * output_width + x] = conv2d_element(input, kernel, kernel_size, input_width, input_height, x, y);
    }
}

// Host function
void conv2d(const torch::Tensor& input, const torch::Tensor& kernel, torch::Tensor& output) {
    const int kernel_size = kernel.size(0);
    const int input_width = input.size(1);
    const int input_height = input.size(0);
    const int output_width = output.size(1);
    const int output_height = output.size(0);

    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks((output_width + threadsPerBlock.x - 1) / threadsPerBlock.x, (output_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(input.data_ptr<float>(), kernel.data_ptr<float>(), output.data_ptr<float>(), kernel_size, input_width, input_height, output_width, output_height);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d", &conv2d, "Modular CUDA 2D convolution");
}