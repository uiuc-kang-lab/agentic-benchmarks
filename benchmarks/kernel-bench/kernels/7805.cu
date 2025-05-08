#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel function to perform 2D convolution
__global__ void conv2d_kernel(const float* __restrict__ input, const float* __restrict__ weight,
                              float* __restrict__ output, int batch_size, int input_channels,
                              int output_channels, int input_height, int input_width,
                              int kernel_height, int kernel_width, int stride, int padding) {
    // Calculate the output position
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_channel = blockIdx.z;
    int batch = blockIdx.w;

    if (out_y >= input_height || out_x >= input_width || out_channel >= output_channels || batch >= batch_size) {
        return;
    }

    float sum = 0.0f;
    for (int in_channel = 0; in_channel < input_channels; ++in_channel) {
        for (int ky = 0; ky < kernel_height; ++ky) {
            int in_y = out_y * stride - padding + ky;
            if (in_y < 0 || in_y >= input_height) continue;

            for (int kx = 0; kx < kernel_width; ++kx) {
                int in_x = out_x * stride - padding + kx;
                if (in_x < 0 || in_x >= input_width) continue;

                int input_idx = ((batch * input_channels + in_channel) * input_height + in_y) * input_width + in_x;
                int weight_idx = ((out_channel * input_channels + in_channel) * kernel_height + ky) * kernel_width + kx;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    int output_idx = ((batch * output_channels + out_channel) * input_height + out_y) * input_width + out_x;
    output[output_idx] = sum;
}

// Host function to launch the kernel
void conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor output,
                 int stride, int padding) {
    const int batch_size = input.size(0);
    const int input_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_channels = weight.size(0);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);

    dim3 blockDim(16, 16);
    dim3 gridDim((input_width + blockDim.x - 1) / blockDim.x,
                 (input_height + blockDim.y - 1) / blockDim.y,
                 output_channels, batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv2d_cuda", ([&] {
        conv2d_kernel<<<gridDim, blockDim>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, input_channels, output_channels,
            input_height, input_width, kernel_height, kernel_width,
            stride, padding);
    }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_cuda", &conv2d_cuda, "Optimized CUDA 2D Convolution");
}