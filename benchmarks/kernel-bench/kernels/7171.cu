#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<int KERNEL_SIZE>
__device__ __forceinline__ float compute_pixel_value(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int n, int oc, int out_y, int out_x,
    int ic, int in_height, int in_width,
    int stride, int padding, int dilation) {
    
    float sum = 0.0f;
    
    #pragma unroll
    for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
        const int in_y = out_y * stride - padding + ky * dilation;
        if (in_y >= 0 && in_y < in_height) {
            #pragma unroll
            for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                const int in_x = out_x * stride - padding + kx * dilation;
                if (in_x >= 0 && in_x < in_width) {
                    const int in_idx = ((n * ic * in_height + in_y) * in_width + in_x);
                    const int w_idx = ((oc * ic * KERNEL_SIZE + ky) * KERNEL_SIZE + kx);
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    return sum;
}

template<int KERNEL_SIZE>
__global__ void conv2d_unrolled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation) {
    
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int oc = blockIdx.z % out_channels;
    const int n = blockIdx.z / out_channels;
    
    if (out_x >= out_width || out_y >= out_height || n >= batch_size) return;
    
    float sum = bias ? bias[oc] : 0.0f;
    
    #pragma unroll 4
    for (int ic = 0; ic < in_channels; ++ic) {
        sum += compute_pixel_value<KERNEL_SIZE>(
            input + n * in_channels * in_height * in_width + ic * in_height * in_width,
            weight + oc * in_channels * KERNEL_SIZE * KERNEL_SIZE + ic * KERNEL_SIZE * KERNEL_SIZE,
            n, oc, out_y, out_x,
            ic, in_height, in_width,
            stride, padding, dilation);
    }
    
    output[((n * out_channels + oc) * out_height + out_y) * out_width + out_x] = sum;
}

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
    if (bias.has_value()) CHECK_INPUT(bias.value());
    
    TORCH_CHECK(groups == 1, "Only groups==1 is supported");
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());
    
    dim3 block(16, 16);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        batch_size * out_channels
    );
    
    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();
    
    switch(kernel_size) {
        case 3:
            conv2d_unrolled_kernel<3><<<grid, block>>>(
                input_ptr, weight_ptr, bias_ptr, output_ptr,
                batch_size, in_channels, out_channels,
                in_height, in_width, out_height, out_width,
                stride, padding, dilation);
            break;
        case 5:
            conv2d_unrolled_kernel<5><<<grid, block>>>(
                input_ptr, weight_ptr, bias_ptr, output_ptr,
                batch_size, in_channels, out_channels,
                in_height, in_width, out_height, out_width,
                stride, padding, dilation);
            break;
        case 7:
            conv2d_unrolled_kernel<7><<<grid, block>>>(
                input_ptr, weight_ptr, bias_ptr, output_ptr,
                batch_size, in_channels, out_channels,
                in_height, in_width, out_height, out_width,
                stride, padding, dilation);
            break;
        default:
            TORCH_CHECK(false, "Unsupported kernel size. Only 3x3, 5x5, and 7x7 kernels are supported.");
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fully unrolled CUDA convolution forward");
}