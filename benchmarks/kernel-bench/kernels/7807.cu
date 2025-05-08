#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Maximum number of floats for the convolution kernel that can be stored in constant memory
#define MAX_KERNEL_SIZE 1024

// Constant memory for storing the convolution kernel weights
__constant__ float constKernel[MAX_KERNEL_SIZE];

// CUDA kernel implementing 2D convolution using constant memory for weights.
// This kernel also supports optional bias addition and group convolution.
__global__ void conv2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias,
    int n,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int out_h,
    int out_w,
    int stride,
    int padding,
    int dilation,
    int groups) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * out_channels * out_h * out_w;
    if (index >= total) return;

    // Calculate output coordinates
    int w_out = index % out_w;
    int h_out = (index / out_w) % out_h;
    int temp = index / (out_w * out_h);
    int c_out = temp % out_channels;
    int b = temp / out_channels;

    // Determine channel grouping
    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group = in_channels / groups;
    int group = c_out / out_channels_per_group;

    float value = 0.0f;
    // Loop over input channels in the appropriate group
    for (int c = 0; c < in_channels_per_group; c++) {
        int in_c = group * in_channels_per_group + c;
        // Loop over the kernel height and width
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int h_in = h_out * stride - padding + kh * dilation;
                int w_in = w_out * stride - padding + kw * dilation;
                if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                    int input_idx = b * (in_channels * in_h * in_w) +
                                    in_c * (in_h * in_w) +
                                    h_in * in_w + w_in;
                    // Calculate weight index: weight is stored as [out_channels, in_channels_per_group, kernel_h, kernel_w]
                    int weight_idx = ((c_out * in_channels_per_group + c) * kernel_h + kh) * kernel_w + kw;
                    value += input[input_idx] * constKernel[weight_idx];
                }
            }
        }
    }

    if (bias != nullptr) {
        value += bias[c_out];
    }

    output[index] = value;
}

// Forward function exposed via PyBind11
// This function copies the weight tensor to constant memory and launches the CUDA kernel.
// It assumes that the weight tensor fits within the constant memory limits.

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

    // Get dimensions from input and weight tensors
    auto n = x.size(0);
    auto in_channels = x.size(1);
    auto in_h = x.size(2);
    auto in_w = x.size(3);

    auto out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int in_channels_per_group = in_channels / groups;
    TORCH_CHECK(weight.size(1) == in_channels_per_group, "Weight shape mismatch with input channels and groups");

    // Calculate output dimensions
    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({n, out_channels, out_h, out_w}, x.options());

    // Ensure the weight tensor fits in constant memory
    int weight_numel = weight.numel();
    TORCH_CHECK(weight_numel <= MAX_KERNEL_SIZE, "Weight tensor too large for constant memory");

    // Copy weight data into constant memory (read-only and cached).
    cudaMemcpyToSymbol(constKernel, weight.data_ptr<float>(), weight_numel * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    int total = n * out_channels * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        (bias.has_value() ? bias.value().data_ptr<float>() : nullptr),
        n, in_channels, in_h, in_w,
        out_channels, kernel_h, kernel_w,
        out_h, out_w,
        stride, padding, dilation,
        groups);

    cudaDeviceSynchronize();
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA 2D Convolution with constant memory usage");
}
