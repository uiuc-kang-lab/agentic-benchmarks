#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Improved depthwise convolution kernel using grid-stride loops for better occupancy
// and minimizing the use of atomic operations. Each thread computes a unique output element,
// thus no atomic operations are required on global memory, which reduces contention.
__global__ void fast_depthwise_conv2d_kernel(
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

    int total = batch * channels * out_h * out_w;
    // Grid-stride loop to cover all output elements
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
        // Decode the flattened index into (b, c, oh, ow)
        int ow = index % out_w;
        int tmp = index / out_w;
        int oh = tmp % out_h;
        tmp = tmp / out_h;
        int c = tmp % channels;
        int b = tmp / channels;

        float sum = 0.0f;
        // Precompute base offsets to reduce repeated arithmetic
        int input_channel_offset = ((b * channels) + c) * in_h * in_w;
        int weight_offset = c * kernel_h; // weight layout: (channels, 1, kernel_h, 1)
        int input_horizontal = ow * stride - padding;  // since kernel width is 1

        // Loop over kernel height dimension
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh * stride - padding + kh * dilation;
            int iw = input_horizontal; 
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int input_idx = input_channel_offset + ih * in_w + iw;
                int weight_idx = weight_offset + kh;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
        // Add bias for the channel
        sum += bias[c];
        int output_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
        output[output_idx] = sum;
    }
}

// Forward function for depthwise 2D convolution
at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    // Ensure contiguous tensors
    x = x.contiguous();
    weight = weight.contiguous();

    int batch = x.size(0);
    int channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int kernel_h = weight.size(2);  // weight shape: (channels, 1, kernel_h, 1)

    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == number of input channels.");
    }

    // Handle optional bias
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
    int threads = (total < 1024) ? total : 1024;
    int blocks = (total + threads - 1) / threads;

    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias_val.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    fast_depthwise_conv2d_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Fast Depthwise 2D Convolution forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
