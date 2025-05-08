#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// 1D kernel with tunable block size via the BLOCK_SIZE macro
// Each thread computes one output element using a flattened index
__global__ void depthwise_conv2d_1d_kernel(
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

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    if (index < total) {
        // Decode the 1D index into (b, c, oh, ow)
        int tmp = index;
        int ow = tmp % out_w;
        tmp /= out_w;
        int oh = tmp % out_h;
        tmp /= out_h;
        int c = tmp % channels;
        int b = tmp / channels;

        float sum = 0.0f;
        // Loop over kernel height dimension (kernel width = 1)
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh * stride - padding + kh * dilation;
            int iw = ow * stride - padding; // since kernel width is 1
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int in_index = ((b * channels + c) * in_h + ih) * in_w + iw;
                sum += __ldg(&input[in_index]) * __ldg(&weight[c * kernel_h + kh]);
            }
        }
        sum += __ldg(&bias[c]);
        int out_index = ((b * channels + c) * out_h + oh) * out_w + ow;
        output[out_index] = sum;
    }
}

// Forward function that uses a 1D grid size. The block size is tunable via the BLOCK_SIZE macro.
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
    int kernel_h = weight.size(2);  // weight shape: (channels, 1, kernel_h, 1)

    // Depthwise convolution requires groups == channels
    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == number of input channels.");
    }

    // Handle bias: if bias is not provided, create a zeros tensor
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

    int total = batch * channels * out_h * out_w;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias_val.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    depthwise_conv2d_1d_kernel<<<blocks, BLOCK_SIZE>>>(
        x_ptr,
        weight_ptr,
        bias_ptr,
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
    m.def("forward", &forward, "Depthwise 2D Convolution forward with tunable block size (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
