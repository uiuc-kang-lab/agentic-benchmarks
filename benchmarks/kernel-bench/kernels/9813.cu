#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Device function: Modular computation of convolution sum for a single output element.
// This function computes the vertical convolution sum for a given output coordinate (oh, ow) using a 1D asymmetric kernel.
__device__ inline float compute_conv(const float* __restrict__ input_channel,
                                       const float* __restrict__ weight_channel,
                                       int oh, int ow,
                                       int in_h, int in_w,
                                       int kernel_h,
                                       int stride, int padding, int dilation) {
    float sum = 0.0f;
    // Compute the common horizontal index
    int iw = ow * stride - padding;
    #pragma unroll
    for (int kh = 0; kh < kernel_h; ++kh) {
        int ih = oh * stride - padding + kh * dilation;
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            int input_idx = ih * in_w + iw;
            sum += input_channel[input_idx] * weight_channel[kh];
        }
    }
    return sum;
}

// CUDA kernel for performing a depthwise 2D convolution using modular device functions
__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
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

    // Determine output coordinates based on thread and block indices
    int ow = threadIdx.x + blockIdx.x * blockDim.x;
    int oh = blockIdx.y % out_h;
    int c  = blockIdx.y / out_h;
    int b  = blockIdx.z;

    if (ow < out_w && c < channels && b < batch) {
        // Pointers for the input and weight corresponding to the current batch and channel
        const float* input_channel = input + ((b * channels + c) * in_h * in_w);
        const float* weight_channel = weight + (c * kernel_h);

        // Modular device function computes the convolution sum.
        float sum = compute_conv(input_channel, weight_channel, oh, ow, in_h, in_w, kernel_h, stride, padding, dilation);
        sum += bias[c];

        int output_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
        output[output_idx] = sum;
    }
}

// Forward function for the CUDA depthwise convolution
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

    // Depthwise convolution requires groups to equal number of channels
    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == number of input channels.");
    }

    // Handle bias: if bias is provided, use it; otherwise initialize to zeros
    at::Tensor bias_val;
    if (bias.has_value() && bias.value().defined()) {
        bias_val = bias.value().contiguous();
    } else {
        bias_val = at::zeros({channels}, x.options());
    }

    // Compute output dimensions
    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - 1) / stride + 1;

    auto output = at::empty({batch, channels, out_h, out_w}, x.options());

    // Launch configuration
    int block_size = 256;
    dim3 threads(block_size, 1, 1);
    dim3 blocks((out_w + block_size - 1) / block_size, channels * out_h, batch);

    depthwise_conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_val.data_ptr<float>(),
        output.data_ptr<float>(),
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
    m.def("forward", &forward, "Depthwise 2D Convolution forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
