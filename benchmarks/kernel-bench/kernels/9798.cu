#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Modular device function to compute the depthwise convolution for one output pixel
__device__ __forceinline__ float compute_conv(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int b,
    int c,
    int oh,
    int ow,
    int in_h,
    int in_w,
    int kernel_h,
    int channels,
    int stride,
    int padding,
    int dilation) {

    float sum = 0.0f;
    // Compute input column index
    int iw = ow * stride - padding;
    // Base index for the (b, c) slice in the input tensor
    int base_input = ((b * channels) + c) * in_h * in_w;
    
    // Loop over the kernel height dimension
    #pragma unroll
    for (int kh = 0; kh < kernel_h; ++kh) {
        int ih = oh * stride - padding + kh * dilation;
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            int input_idx = base_input + ih * in_w + iw;
            int weight_idx = c * kernel_h + kh;
            sum += input[input_idx] * weight[weight_idx];
        }
    }
    return sum;
}

// CUDA kernel employing the modular device function
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

    // Determine the output column index
    int ow = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Use blockIdx.y to cover both output row (oh) and channel (c)
    int linear_idx = blockIdx.y;
    int oh = linear_idx % out_h;
    int c  = linear_idx / out_h;
    
    // Use blockIdx.z for batch index
    int b = blockIdx.z;

    if (ow < out_w && c < channels && b < batch) {
        float sum = compute_conv(input, weight, b, c, oh, ow, in_h, in_w, kernel_h, channels, stride, padding, dilation);
        sum += bias[c];
        int output_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
        output[output_idx] = sum;
    }
}

// Forward function for the modular implementation of Depthwise 2D Convolution
at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    x = x.contiguous();
    weight = weight.contiguous();

    int batch = x.size(0);
    int channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int kernel_h = weight.size(2);  // weight shape assumed as (channels, 1, kernel_h, 1)

    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == number of input channels.");
    }

    at::Tensor bias_val;
    if (bias.has_value() && bias.value().defined()) {
        bias_val = bias.value().contiguous();
    } else {
        bias_val = at::zeros({channels}, x.options());
    }

    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - 1) / stride + 1;

    auto output = at::empty({batch, channels, out_h, out_w}, x.options());

    // Configure grid and block dimensions
    dim3 threads(256, 1, 1);
    // blockIdx.x covers output columns, blockIdx.y covers (channel, output row), blockIdx.z covers batch
    dim3 blocks((out_w + threads.x - 1) / threads.x, channels * out_h, batch);

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
    m.def("forward", &forward, "Modular Depthwise 2D Convolution forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
