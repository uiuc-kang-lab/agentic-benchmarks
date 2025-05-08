#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This CUDA kernel performs 2D convolution on asymmetric input with a square kernel.
// It optimizes thread and block distribution to ensure even workload across threads.
// It also includes a fallback to native PyTorch conv2d for unsupported cases.

__global__ void conv2d_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    int n = blockIdx.x;       // batch index
    int oc = blockIdx.y;      // output channel index
    int out_y = blockIdx.z * blockDim.y + threadIdx.y;
    int out_x = threadIdx.x;

    if (out_y >= out_height || out_x >= out_width) return;

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_y = out_y * stride - padding + ky * dilation;
                int in_x = out_x * stride - padding + kx * dilation;

                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int input_idx = n * in_channels * in_height * in_width
                                  + ic * in_height * in_width
                                  + in_y * in_width
                                  + in_x;
                    int weight_idx = oc * in_channels * kernel_size * kernel_size
                                   + ic * kernel_size * kernel_size
                                   + ky * kernel_size
                                   + kx;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (bias) {
        sum += bias[oc];
    }
    
    int output_idx = n * out_channels * out_height * out_width
                   + oc * out_height * out_width
                   + out_y * out_width
                   + out_x;
    output[output_idx] = sum;
}

// Host function that prepares the tensors and launches the CUDA kernel
// Falls back to native PyTorch conv2d for unsupported cases

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

    // Check if groups is supported
    if (groups != 1) {
        if (bias.has_value()) {
            return torch::conv2d(x, weight, bias.value(), {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
        } else {
            return torch::conv2d(x, weight, torch::Tensor(), {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
        }
    }

    // Extract dimensions
    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);

    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());
    
    dim3 threads(out_width, 1);
    dim3 blocks(batch, out_channels, (out_height + threads.y - 1) / threads.y);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = (bias.has_value()) ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    conv2d_optimized_kernel<<<blocks, threads>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        batch, in_channels, out_channels,
        in_height, in_width, out_height, out_width,
        kernel_size, stride, padding, dilation);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Efficient CUDA forward function for 2D convolution with optimized and fallback support");
}