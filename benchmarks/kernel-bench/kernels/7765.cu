#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_shared_memory_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int batch_size,
    int input_channels,
    int output_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int padding) {

    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int input_row_start = blockIdx.y * stride - padding;
    int input_col_start = blockIdx.x * stride - padding;

    // Loading input tile to shared memory
    for (int c = 0; c < input_channels; ++c) {
        for (int dy = ty; dy < kernel_height; dy += blockDim.y) {
            for (int dx = tx; dx < kernel_width; dx += blockDim.x) {
                int in_y = input_row_start + dy;
                int in_x = input_col_start + dx;
                if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                    shared_input[(c * kernel_height + dy) * kernel_width + dx] = input[(blockIdx.z * input_channels + c) * input_height * input_width + in_y * input_width + in_x];
                } else {
                    shared_input[(c * kernel_height + dy) * kernel_width + dx] = 0;
                }
            }
        }
    }

    __syncthreads();

    // Convolution operation
    for (int out_c = 0; out_c < output_channels; ++out_c) {
        float result = 0.0f;
        for (int c = 0; c < input_channels; ++c) {
            for (int dy = 0; dy < kernel_height; ++dy) {
                for (int dx = 0; dx < kernel_width; ++dx) {
                    result += shared_input[(c * kernel_height + dy) * kernel_width + dx] * kernel[((out_c * input_channels + c) * kernel_height + dy) * kernel_width + dx];
                }
            }
        }

        int output_index = ((blockIdx.z * output_channels + out_c) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
        output[output_index] = result;
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

    int batch_size = x.size(0);
    int input_channels = x.size(1);
    int input_height = x.size(2);
    int input_width = x.size(3);
    int output_channels = weight.size(0);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    int output_height = (input_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    int output_width = (input_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor output = torch::empty({batch_size, output_channels, output_height, output_width}, options);

    // Define block and grid sizes
    dim3 block_dim(16, 16);
    dim3 grid_dim((output_width + block_dim.x - 1) / block_dim.x, (output_height + block_dim.y - 1) / block_dim.y, batch_size);

    // Launch kernel
    int shared_memory_size = input_channels * kernel_height * kernel_width * sizeof(float);
    conv2d_shared_memory_kernel<<<grid_dim, block_dim, shared_memory_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_channels,
        output_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride,
        padding);

    // Synchronization to ensure completion before returning
    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA 2D Convolution with Shared Memory");
}