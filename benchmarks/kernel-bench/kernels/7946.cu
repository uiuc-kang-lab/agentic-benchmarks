#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel function using efficient thread and block mapping
__global__ void conv2d_kernel(float *x, float *weight, float *output, 
                              int stride, int padding, int dilation, 
                              int in_channels, int out_channels,
                              int input_size, int kernel_size, int output_size) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int c_out = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (out_x < output_size && out_y < output_size && c_out < out_channels) {
        // Each thread computes one element in the output feature map
        float result = 0.0f;
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                for (int ky = 0; ky < kernel_size; ++ky) {
                    int in_x = out_x * stride + kx - padding;
                    int in_y = out_y * stride + ky - padding;
                    if (in_x >= 0 && in_x < input_size && in_y >= 0 && in_y < input_size) {
                        float value = x[((c_in * input_size + in_x) * input_size + in_y)];
                        float filter = weight[((c_out * in_channels + c_in) * kernel_size + kx) * kernel_size + ky];
                        result += value * filter;
                    }
                }
            }
        }
        output[(c_out * output_size + out_y) * output_size + out_x] = result;
    }
}

void optimized_conv2d(torch::Tensor x, torch::Tensor weight, torch::Tensor output,
                      int stride, int padding, int dilation) {
    const auto input_size = x.size(2);
    const auto kernel_size = weight.size(2);
    const auto in_channels = x.size(1);
    const auto out_channels = weight.size(0);
    const auto output_size = (input_size + 2 * padding - kernel_size) / stride + 1;

    dim3 block(16, 16, 1);
    dim3 grid((output_size + block.x - 1) / block.x, 
              (output_size + block.y - 1) / block.y, 
              (out_channels + block.z - 1) / block.z);

    conv2d_kernel<<<grid, block>>>(x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), 
                                   stride, padding, dilation, in_channels, out_channels, 
                                   input_size, kernel_size, output_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_conv2d, "Optimized CUDA forward function for 2D convolution with improved thread mapping");
}