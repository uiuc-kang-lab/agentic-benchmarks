#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA kernel for performing a depthwise 2D convolution with an asymmetric kernel
// (kernel height > 1, kernel width = 1)
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
    int dilation)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    if (index < total) {
        // Decode the flattened index into 4D coordinates: (b, c, oh, ow)
        int ow = index % out_w;
        int tmp = index / out_w;
        int oh = tmp % out_h;
        tmp = tmp / out_h;
        int c = tmp % channels;
        int b = tmp / channels;

        float sum = 0.f;
        // Loop over kernel height dimension
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh * stride - padding + kh * dilation;
            // Since kernel width is 1, the input column is computed as:
            int iw = ow * stride - padding;
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                // weight shape: (channels, 1, kernel_h, 1) => index as (c, kh)
                int weight_idx = c * kernel_h + kh;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
        // Add bias for the current channel
        sum += bias[c];
        int output_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
        output[output_idx] = sum;
    }
}

// The forward function now accepts bias as an optional tensor.
// If bias is None, a zero bias tensor will be used.
at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups)
{
    // Ensure the inputs are contiguous.
    x = x.contiguous();
    weight = weight.contiguous();

    // Retrieve input dimensions.
    int batch = x.size(0);
    int channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int kernel_h = weight.size(2);  // weight shape: (channels, 1, kernel_h, 1)
    
    // For depthwise convolution, groups should equal the number of channels.
    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == number of input channels.");
    }

    // Handle bias: if bias is None, create a zeros tensor.
    at::Tensor bias_val;
    if (bias.has_value() && bias.value().defined()) {
        bias_val = bias.value().contiguous();
    } else {
        bias_val = at::zeros({channels}, x.options());
    }

    // Compute output dimensions.
    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - 1) / stride + 1;

    // Allocate output tensor.
    auto output = at::empty({batch, channels, out_h, out_w}, x.options());

    // Set up CUDA launch parameters.
    int total = batch * channels * out_h * out_w;
    int threads = 1024;
    int blocks = (total + threads - 1) / threads;

    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias_val.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Launch the CUDA kernel.
    depthwise_conv2d_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}