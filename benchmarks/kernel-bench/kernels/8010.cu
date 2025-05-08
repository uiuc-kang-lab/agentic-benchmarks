#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv_transposed1d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t in_width,
    int64_t out_width,
    int64_t stride,
    int64_t padding,
    int64_t groups,
    int64_t group_stride) {
    
    const int64_t g = blockIdx.y;
    const int64_t oc = blockIdx.z;
    
    for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
         pos < out_width;
         pos += gridDim.x * blockDim.x) {
        
        float total = 0.0f;
        const int64_t group_offset = g * group_stride;
        
        for (int64_t ic = 0; ic < in_channels/groups; ++ic) {
            for (int64_t k = 0; k < kernel_size; ++k) {
                const int64_t input_pos = (pos - k + padding) / stride;
                if ((pos - k + padding) %% stride == 0 && input_pos >= 0 && input_pos < in_width) {
                    const float* w_ptr = weight + oc * (in_channels/groups) * kernel_size + ic * kernel_size + k;
                    const float* x_ptr = input + (group_offset + ic) * in_width + input_pos;
                    total += (*x_ptr) * (*w_ptr);
                }
            }
        }
        
        float* out_ptr = output + (g * out_channels/groups + oc) * out_width + pos;
        if (bias) {
            *out_ptr = total + bias[g * out_channels/groups + oc];
        } else {
            *out_ptr = total;
        }
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
    
    const int64_t in_channels = x.size(1);
    const int64_t in_width = x.size(2);
    const int64_t kernel_size = weight.size(3);
    const int64_t out_channels = weight.size(1) * groups;
    
    const int64_t out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    auto output = torch::empty({x.size(0), out_channels, out_width}, x.options());
    
    const int64_t group_stride = in_channels / groups;
    
    dim3 blocks(
        (out_width + 255) / 256,
        groups,
        weight.size(1)
    );
    
    conv_transposed1d_kernel<<<blocks, 256>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        in_channels,
        out_channels,
        kernel_size,
        in_width,
        out_width,
        stride,
        padding,
        groups,
        group_stride
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized transposed 1D convolution forward (CUDA)");
}