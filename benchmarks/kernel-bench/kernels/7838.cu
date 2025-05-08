#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Macros for checking tensor properties
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define the number of threads per block. Experiment with 32, 64, 128, 256, 512.
// For example, setting to 512 for high occupancy on modern GPUs.
#define THREADS_PER_BLOCK 512

// A simple CUDA kernel using a strided loop over output elements.
// Each thread computes multiple output elements in steps of (gridDim.x * blockDim.x).

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_height,
    const int kernel_width,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * out_channels * out_height * out_width;
    const int stride_size = gridDim.x * blockDim.x;

    // Process multiple output elements per thread
    for (int pos = idx; pos < total; pos += stride_size) {
        // Compute output coordinates from linear index
        int w_out = pos % out_width;
        int h_out = (pos / out_width) % out_height;
        int c_out = (pos / (out_width * out_height)) % out_channels;
        int b = pos / (out_width * out_height * out_channels);

        // Initialize the accumulator with bias if provided
        float sum = bias ? bias[c_out] : 0.0f;

        // Iterate over all input channels and apply the kernel
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int h_in = h_out * stride - padding + kh;
                    int w_in = w_out * stride - padding + kw;

                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                        int input_idx = ((b * in_channels + c_in) * in_height + h_in) * in_width + w_in;
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_height + kh) * kernel_width + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        int out_idx = ((b * out_channels + c_out) * out_height + h_out) * out_width + w_out;
        output[out_idx] = sum;
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

    // Fall back to PyTorch's optimized conv2d if non-standard parameters are used
    if (dilation != 1 || groups != 1) {
        return torch::conv2d(x, weight, bias.has_value() ? bias.value() : torch::Tensor(),
                             {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    int out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_width) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, x.options());

    int total = batch_size * out_channels * out_height * out_width;
    int blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    conv2d_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_height,
        kernel_width,
        out_height,
        out_width,
        stride,
        padding);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA 2D Convolution with Tunable Block Size");
}
