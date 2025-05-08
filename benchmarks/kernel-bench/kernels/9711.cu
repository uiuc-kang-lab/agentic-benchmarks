#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Define maximum sizes for constant memory arrays (in number of floats)
#define MAX_CONST_WEIGHT_SIZE 16384   // 64KB (16384 * 4 bytes)
#define MAX_CONST_BIAS_SIZE 4096        // 16KB (4096 * 4 bytes)

// Constant memory for weights and bias
__constant__ float d_weight[MAX_CONST_WEIGHT_SIZE];
__constant__ float d_bias[MAX_CONST_BIAS_SIZE];

// Kernel using constant memory for depthwise convolution weights and bias
__global__ void depthwise_conv2d_kernel_const(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int stride,
    int padding,
    int dilation) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    if (index < total) {
        // Decode flattened index into (b, c, oh, ow)
        int ow = index % out_w;
        int tmp = index / out_w;
        int oh = tmp % out_h;
        tmp = tmp / out_h;
        int c = tmp % channels;
        int b = tmp / channels;

        float sum = 0.f;
        #pragma unroll
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh * stride - padding + kh * dilation;
            int iw = ow * stride - padding; // As kernel width is 1
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                // Access weight from constant memory
                sum += input[input_idx] * d_weight[c * kernel_h + kh];
            }
        }
        // Add bias from constant memory
        sum += d_bias[c];
        int out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
        output[out_idx] = sum;
    }
}

// Forward function for depthwise convolution using constant memory
at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    // Ensure inputs are contiguous
    x = x.contiguous();
    weight = weight.contiguous();

    int batch = x.size(0);
    int channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int kernel_h = weight.size(2); // weight shape: (channels, 1, kernel_h, 1)

    // Depthwise convolution requires groups == number of channels
    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == number of input channels.");
    }

    // Handle bias: if provided, use it; else, create zeros
    at::Tensor bias_val;
    if (bias.has_value() && bias.value().defined()) {
        bias_val = bias.value().contiguous();
    } else {
        bias_val = at::zeros({channels}, x.options());
    }

    // Ensure the weight and bias sizes fit into constant memory
    if (channels * kernel_h > MAX_CONST_WEIGHT_SIZE) {
        throw std::invalid_argument("Weight tensor size exceeds constant memory limits.");
    }
    if (channels > MAX_CONST_BIAS_SIZE) {
        throw std::invalid_argument("Bias tensor size exceeds constant memory limits.");
    }

    // Copy weight and bias to constant memory
    cudaMemcpyToSymbol(d_weight, weight.data_ptr<float>(), channels * kernel_h * sizeof(float));
    cudaMemcpyToSymbol(d_bias, bias_val.data_ptr<float>(), channels * sizeof(float));

    // Compute output dimensions
    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - 1) / stride + 1;
    
    auto output = at::empty({batch, channels, out_h, out_w}, x.options());

    int total = batch * channels * out_h * out_w;
    int threads = 1024;
    int blocks = (total + threads - 1) / threads;

    const float* x_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    depthwise_conv2d_kernel_const<<<blocks, threads>>>(
        x_ptr,
        output_ptr,
        batch,
        channels,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h,
        stride,
        padding,
        dilation
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution forward with constant memory (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
