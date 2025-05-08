#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel leverages __ldg() for read-only memory accesses (input, weight, and bias) to optimize global memory loads.
// It also organizes thread and block mapping to encourage 128-bit aligned accesses when possible. 
// Each thread computes one output pixel for all output channels, with proper boundary and dilation handling.

__global__ void conv2d_ldg_aligned_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int kernel_size,
    int output_h,
    int output_w,
    int stride,
    int padding,
    int dilation) {

    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ow < output_w && oh < output_h && b < batch) {
        // Loop over output channels
        for (int oc = 0; oc < out_channels; oc++) {
            float sum = 0.0f;
            if (bias) {
                sum = __ldg(&bias[oc]);
            }
            // Convolution summing over input channels and kernel window
            for (int ic = 0; ic < in_channels; ic++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int ih = oh * stride - padding + kh * dilation;
                        int iw = ow * stride - padding + kw * dilation;
                        if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                            int input_idx = ((b * in_channels + ic) * input_h + ih) * input_w + iw;
                            int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                            sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                        }
                    }
                }
            }
            int output_idx = ((b * out_channels + oc) * output_h + oh) * output_w + ow;
            output[output_idx] = sum;
        }
    }
}

// The forward function sets up the kernel dimensions and calls the optimized convolution kernel.
// It uses __ldg() to ensure read-only data is loaded efficiently and assumes that the input pointers are aligned to 128-bit boundaries for best performance.

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }
    TORCH_CHECK(groups == 1, "Only groups==1 is supported.");

    int batch = x.size(0);
    int in_channels = x.size(1);
    int input_h = x.size(2);
    int input_w = x.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // assuming square kernel

    int output_h = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_w = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch, out_channels, output_h, output_w}, x.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((output_w + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (output_h + BLOCK_SIZE - 1) / BLOCK_SIZE,
                batch);

    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    conv2d_ldg_aligned_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        input_h,
        input_w,
        kernel_size,
        output_h,
        output_w,
        stride,
        padding,
        dilation);

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution utilizing __ldg() and aligned memory accesses");
}
