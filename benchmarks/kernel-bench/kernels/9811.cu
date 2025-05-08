#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA kernel using a grid-stride loop for even workload distribution
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

    int total = batch * channels * out_h * out_w;
    
    // Grid-stride loop to distribute work evenly
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total;
         index += blockDim.x * gridDim.x) {

        // Decode the flattened index into 4D coordinates: (b, c, oh, ow)
        int ow = index % out_w;
        int tmp = index / out_w;
        int oh = tmp % out_h;
        tmp = tmp / out_h;
        int c = tmp % channels;
        int b = tmp / channels;

        float sum = 0.0f;

        // Cache weights and bias for the current channel in registers
        // Assumption: kernel_h is small (e.g., <= 16)
        float w_cache[16];
        #pragma unroll
        for (int i = 0; i < kernel_h; i++) {
            w_cache[i] = __ldg(&weight[c * kernel_h + i]);
        }
        float bias_val = __ldg(&bias[c]);

        #pragma unroll
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh * stride - padding + kh * dilation;
            // Asymmetric kernel width is 1
            int iw = ow * stride - padding;
            
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                sum += input[input_idx] * w_cache[kh];
            }
        }

        sum += bias_val;
        int output_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
        output[output_idx] = sum;
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

    // Ensure inputs are contiguous
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

    at::Tensor bias_val;
    if (bias.has_value() && bias.value().defined()) {
        bias_val = bias.value().contiguous();
    } else {
        bias_val = at::zeros({channels}, x.options());
    }

    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - 1) / stride + 1;

    auto output = at::empty({batch, channels, out_h, out_w}, x.options());

    int total = batch * channels * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

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
