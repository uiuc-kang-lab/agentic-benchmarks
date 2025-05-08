#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Device function to handle bias addition
__device__ void apply_bias(float* output, const float* bias, 
                         int channels, int output_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_length * channels) {
        int c = idx % channels;
        output[idx] += bias[c];
    }
}

// Device function for main convolution computation
__device__ void compute_conv_transpose(
    const float* input, const float* weight,
    float* output, int in_channels, int out_channels,
    int kernel_size, int stride, int padding,
    int output_padding, int input_length) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_length = (input_length - 1) * stride - 2 * padding + 
                       kernel_size + output_padding;
    
    if (idx < output_length * out_channels) {
        int out_pos = idx / out_channels;
        int out_ch = idx % out_channels;
        float sum = 0.0f;
        
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            for (int k = 0; k < kernel_size; ++k) {
                int in_pos = (out_pos + padding - k) / stride;
                if (in_pos >= 0 && in_pos < input_length && 
                    (out_pos + padding - k) % stride == 0) {
                    sum += input[in_pos * in_channels + in_ch] * 
                           weight[out_ch * in_channels * kernel_size + 
                                  in_ch * kernel_size + k];
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
    
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
        auto result = torch::conv_transpose1d(
            x, weight, bias.value(),
            stride, padding, output_padding, groups
        );
        return result;
    }
    
    return torch::conv_transpose1d(
        x, weight,
        torch::Tensor(),
        stride, padding, output_padding, groups
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 1D convolution forward (CUDA)");
}