#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Device function to compute input indices
__device__ __forceinline__ void compute_indices(
    int index, int out_w, int out_h, int channels,
    int& b, int& c, int& oh, int& ow) {
    ow = index % out_w;
    int tmp = index / out_w;
    oh = tmp % out_h;
    tmp = tmp / out_h;
    c = tmp % channels;
    b = tmp / channels;
}

// Device function to check bounds
__device__ __forceinline__ bool is_within_bounds(
    int ih, int iw, int in_h, int in_w) {
    return (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w);
}

// Device function to compute input offset
__device__ __forceinline__ int compute_input_offset(
    int b, int c, int ih, int iw,
    int channels, int in_h, int in_w) {
    return ((b * channels + c) * in_h + ih) * in_w + iw;
}

// Device function to perform the actual convolution computation
__device__ __forceinline__ float compute_conv_result(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int b, int c, int oh, int ow,
    int in_h, int in_w, int kernel_h,
    int stride, int padding, int dilation,
    int channels) {
    
    float sum = 0.0f;
    #pragma unroll
    for (int kh = 0; kh < kernel_h; ++kh) {
        int ih = oh * stride - padding + kh * dilation;
        int iw = ow * stride - padding;
        
        if (is_within_bounds(ih, iw, in_h, in_w)) {
            int input_idx = compute_input_offset(b, c, ih, iw, channels, in_h, in_w);
            int weight_idx = c * kernel_h + kh;
            sum += input[input_idx] * weight[weight_idx];
        }
    }
    return sum;
}

__global__ void modular_depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int kernel_h, int stride,
    int padding, int dilation) {
    
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * channels * out_h * out_w;
    
    if (index < total) {
        int b, c, oh, ow;
        compute_indices(index, out_w, out_h, channels, b, c, oh, ow);
        
        float result = compute_conv_result(
            input, weight, b, c, oh, ow,
            in_h, in_w, kernel_h,
            stride, padding, dilation, channels);
        
        result += bias[c];
        output[index] = result;
    }
}

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
    int kernel_h = weight.size(2);

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

    const int threads = 256;
    const int total = batch * channels * out_h * out_w;
    const int blocks = (total + threads - 1) / threads;

    modular_depthwise_conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_val.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels,
        in_h, in_w,
        out_h, out_w,
        kernel_h, stride,
        padding, dilation
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