#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Device function for convolution operation
__device__ float conv2d_kernel(const float* input, const float* weight, int kernel_size, int stride, int padding, 
                             int input_width, int input_height, int input_channels, int output_x, int output_y, int channel) {
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int in_x = stride * output_x + j - padding;
            int in_y = stride * output_y + i - padding;
            
            if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                int input_idx = (channel * input_height * input_width) + (in_y * input_width + in_x);
                int weight_idx = (channel * kernel_size * kernel_size) + (i * kernel_size + j);
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    return sum;
}

// Kernel function for applying convolution across the entire image
__global__ void conv2d_forward_kernel(const float* input, const float* weight, float* output, int kernel_size, int stride, int padding, int input_width, int input_height, int output_width, int output_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_width && y < output_height) {
        output[y * output_width + x] = conv2d_kernel(input, weight, kernel_size, stride, padding, input_width, input_height, x, y);
    }
}

void forward_conv2d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    int stride,
    int padding
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);

    int input_height = input.size(0);
    int input_width = input.size(1);
    int kernel_size = weight.size(0);
    int output_height = output.size(0);
    int output_width = output.size(1);

    const int threads = 16;
    const dim3 blocks((output_width + threads - 1) / threads, (output_height + threads - 1) / threads);
    const dim3 threads_per_block(threads, threads);

    conv2d_forward_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel_size,
        stride,
        padding,
        input_width,
        input_height,
        output_width,
        output_height
    );

    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_conv2d, "Modular CUDA forward function for 2D convolution");
}