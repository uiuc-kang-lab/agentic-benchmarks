#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    if (idx < batch_size * out_channels * output_length) {
        const int o = idx % output_length;
        const int c = (idx / output_length) % out_channels;
        const int b = idx / (output_length * out_channels);
        
        float sum = bias ? __ldg(&bias[c]) : 0.0f;
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int k = 0; k < kernel_size; k++) {
                const int input_idx = (o + padding - k) / stride;
                if ((o + padding - k) % stride == 0 && input_idx >= 0 && input_idx < input_length) {
                    sum += __ldg(&input[b * in_channels * input_length + ic * input_length + input_idx]) *
                           __ldg(&weight[c * in_channels * kernel_size + ic * kernel_size + k]);
                }
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) CHECK_INPUT(bias.value());
    
    auto input_size = x.sizes();
    auto weight_size = weight.sizes();
    
    const int batch_size = input_size[0];
    const int in_channels = input_size[1];
    const int input_length = input_size[2];
    const int out_channels = weight_size[1] * groups;
    const int kernel_size = weight_size[2];
    
    const int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, output_length},
                              torch::device(torch::kCUDA).dtype(torch::kFloat32));
    
    const int threads = 256;
    const int blocks = (batch_size * out_channels * output_length + threads - 1) / threads;
    
    conv_transpose1d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        output_padding);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 1D convolution forward (CUDA)");
}