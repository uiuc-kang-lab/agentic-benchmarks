#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Optimized kernel that leverages __ldg() for read-only global memory accesses and
// assumes that the pointers provided by PyTorch are aligned to 128-bit boundaries.
// This minimizes memory latency by using the read-only data cache for input, weight, and bias.
__global__ void depthwise_conv2d_kernel_aligned_ldg(
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
        // Decode the flattened index into 4D coordinates: (b, c, oh, ow)
        int ow = index % out_w;
        int tmp = index / out_w;
        int oh = tmp % out_h;
        tmp = tmp / out_h;
        int c = tmp % channels;
        int b = tmp / channels;

        float sum = 0.f;
        // Unroll the kernel height loop for reduced overhead
        #pragma unroll
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh * stride - padding + kh * dilation;
            int iw = ow * stride - padding;  // kernel width is 1
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                int weight_idx = c * kernel_h + kh;  // weight shape: (channels, 1, kernel_h, 1)
                float in_val = __ldg(&input[input_idx]);
                float w_val = __ldg(&weight[weight_idx]);
                sum += in_val * w_val;
            }
        }
        sum += __ldg(&bias[c]);
        int out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
        output[out_idx] = sum;
    }
}

// Forward function for the depthwise convolution op. It ensures inputs are contiguous and
// sets up the CUDA kernel launch parameters.
at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    // Ensure the inputs are contiguous.
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

    // Handle bias: if bias is not provided, use a zeros tensor
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
    int threads = 1024;
    int blocks = (total + threads - 1) / threads;

    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias_val.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Launch the optimized kernel
    depthwise_conv2d_kernel_aligned_ldg<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution forward with aligned __ldg loads (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
