#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Kernel that uses a flat grid-stride loop over the entire output tensor
// to ensure an even distribution of work across threads and blocks.
// Each thread computes one or more output elements by decoding a flattened index
// into (batch, channel, out_h, out_w) coordinates. __ldg() is used for read-only memory accesses.
__global__ void depthwise_conv2d_flat_grid(
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

    // Total number of output elements
    int total = batch * channels * out_h * out_w;

    // Grid-stride loop to distribute outputs evenly over all threads
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        // Decode the flattened index into (b, c, oh, ow)
        int rem = idx;
        int ow = rem % out_w;
        rem /= out_w;
        int oh = rem % out_h;
        rem /= out_h;
        int c = rem % channels;
        int b = rem / channels;

        float sum = 0.f;
        #pragma unroll
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh * stride - padding + kh * dilation;
            int iw = ow * stride - padding;  // kernel width is assumed to be 1
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int in_index = ((b * channels + c) * in_h + ih) * in_w + iw;
                float in_val = __ldg(&input[in_index]);
                int w_index = c * kernel_h + kh;
                float w_val = __ldg(&weight[w_index]);
                sum += in_val * w_val;
            }
        }
        sum += __ldg(&bias[c]);
        output[idx] = sum;
    }
}

// Forward function: prepares tensors and launches the kernel with a flat grid-stride loop
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

    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias_val.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Launch kernel using a flat grid-stride loop to evenly distribute work
    depthwise_conv2d_flat_grid<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution forward with flat grid-stride loop (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
