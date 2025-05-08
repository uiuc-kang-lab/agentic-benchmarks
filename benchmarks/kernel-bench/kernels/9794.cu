#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// This kernel uses a grid configuration that maps each thread to a unique output column
// for a given batch, channel, and output row. By aligning threads along the width
// dimension, consecutive threads access consecutive memory addresses in output, ensuring
// memory coalescing. Additionally, __ldg is used to fetch read-only data from input,
// weight, and bias arrays to leverage the read-only cache on NVIDIA GPUs.
__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int kernel_h,
    int stride,
    int padding,
    int dilation) {

    // Compute output column index (ow) for this thread
    int ow = blockIdx.x * blockDim.x + threadIdx.x;

    // Combine batch, channel, and output row into one index
    // blockIdx.y ranges over (batch * channels * out_h)
    int idx = blockIdx.y;
    int oh = idx % out_h;
    int tmp = idx / out_h;
    int c = tmp % channels;
    int b = tmp / channels;

    if (ow < out_w) {
        float sum = 0.f;
        // Compute base indices for input and output for this (b, c)
        int input_base = ((b * channels + c) * in_h) * in_w;
        int output_base = ((b * channels + c) * out_h) * out_w;

        // Compute starting coordinates in input considering stride and padding
        int in_col = ow * stride - padding;
        int in_row_base = oh * stride - padding;

        // Iterate over the kernel height
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = in_row_base + kh * dilation;
            int iw = in_col; // kernel width is 1
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int input_idx = input_base + ih * in_w + iw;
                int weight_idx = c * kernel_h + kh * 1; // weight layout: (channels, 1, kernel_h, 1)
                sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
            }
        }

        // Add bias
        sum += __ldg(&bias[c]);

        // Write result to output (coalesced write across ow dimension)
        int output_idx = output_base + oh * out_w + ow;
        output[output_idx] = sum;
    }
}

// Forward function for the depthwise convolution
at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) 
{
    // Ensure contiguous memory layouts
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

    // Configure grid and block dimensions to ensure coalesced memory accesses
    dim3 threads(256, 1, 1);
    // Each block in the x-dimension processes a chunk of output columns,
    // while blockIdx.y covers each (batch, channel, out_row) combination.
    dim3 blocks((out_w + threads.x - 1) / threads.x, batch * channels * out_h, 1);

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
