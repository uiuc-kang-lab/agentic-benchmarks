#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// This kernel splits the convolution work across threads by mapping one thread per partial product
// for each output pixel, i.e. one thread for each combination of (b, c, oh, ow, kh).
// Each thread computes one product and uses atomicAdd to accumulate into the output pixel.
// To add the bias exactly once per output, the thread where kh == 0 adds the bias along with its product.
// Atomic operations are used only to resolve the race condition on output accumulation. Since the number
// of atomic adds per output pixel is small (kernel_h, typically 3 or 5), this minimizes global memory contention.

__global__ void depthwise_conv2d_atomic_kernel(
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

    // total threads: one per (b, c, oh, ow, kh)
    int total = batch * channels * out_h * out_w * kernel_h;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < total; idx += blockDim.x * gridDim.x) {
        // Decode multi-index: first, kh dimension
        int kh = idx % kernel_h;
        int temp = idx / kernel_h;  // now temp corresponds to (b, c, oh, ow)
        int ow = temp % out_w;
        temp /= out_w;
        int oh = temp % out_h;
        temp /= out_h;
        int c = temp % channels;
        int b = temp / channels;

        // Compute the input coordinate
        int ih = oh * stride - padding + kh * dilation;
        int iw = ow * stride - padding; // kernel width is 1

        float prod = 0.0f;
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
            int weight_idx = c * kernel_h + kh; // weight shape: (channels, 1, kernel_h, 1)
            prod = input[input_idx] * weight[weight_idx];
        }

        // Calculate the output index (each output pixel gets contributions from kernel_h threads)
        int out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;

        // For the thread corresponding to kh == 0, add bias along with its product
        if (kh == 0) {
            prod += bias[c];
        }

        // Accumulate the partial product using atomicAdd
        atomicAdd(&output[out_idx], prod);
    }
}

// Forward function for depthwise 2D convolution using atomic accumulation
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

    // Handle bias; if bias is not provided, create a zero tensor
    at::Tensor bias_val;
    if (bias.has_value() && bias.value().defined()) {
        bias_val = bias.value().contiguous();
    } else {
        bias_val = at::zeros({channels}, x.options());
    }

    // Calculate output dimensions
    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - 1) / stride + 1;

    // IMPORTANT: Allocate output tensor and initialize to 0, since we use atomic adds.
    auto output = at::zeros({batch, channels, out_h, out_w}, x.options());

    // Total number of threads: one for each partial product
    int total = batch * channels * out_h * out_w * kernel_h;
    int threads = 256;  // Adjusted block size for better occupancy balance
    int blocks = (total + threads - 1) / threads;

    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias_val.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    depthwise_conv2d_atomic_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution forward with atomic accumulation (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
