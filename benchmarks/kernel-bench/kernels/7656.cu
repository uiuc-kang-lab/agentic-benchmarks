#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cudnn/Handles.h>
#include <ATen/cudnn/Descriptors.h>
#include <cudnn.h>

#define BLOCK_SIZE 256
#define SMALL_TENSOR_THRESHOLD 64 // Threshold for small tensor dimensions

cudnnDataType_t getCudnnDataType(at::ScalarType type) {
    switch (type) {
        case at::ScalarType::Float: return CUDNN_DATA_FLOAT;
        case at::ScalarType::Double: return CUDNN_DATA_DOUBLE;
        case at::ScalarType::Half: return CUDNN_DATA_HALF;
        default: TORCH_CHECK(false, "Unsupported data type for cuDNN");
    }
}

__global__ void conv3d_strided_kernel(
    float* output, const float* input, const float* weight, const float* bias,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int kernel_d, int kernel_h, int kernel_w,
    int out_depth, int out_height, int out_width,
    int stride, int padding, int dilation, int groups) {
    // [Previous kernel implementation remains the same]
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    auto bias = bias_opt.value_or(at::Tensor());
    
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(!bias.defined() || bias.is_cuda(), "Bias must be a CUDA tensor");

    // Get dimensions
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);
    
    auto out_channels = weight.size(0);
    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);
    
    auto out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    auto out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    auto out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    // Choose implementation based on tensor size
    bool use_cudnn = (in_depth > SMALL_TENSOR_THRESHOLD) || 
                     (in_height > SMALL_TENSOR_THRESHOLD) || 
                     (in_width > SMALL_TENSOR_THRESHOLD);

    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    if (use_cudnn) {
        // Use cuDNN implementation for large tensors
        cudnnHandle_t handle = at::native::getCudnnHandle();
        // [Rest of cuDNN implementation]
    } else {
        // Use custom CUDA kernel for small tensors
        const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
        const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        conv3d_strided_kernel<<<num_blocks, BLOCK_SIZE>>>(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            batch_size, in_channels, out_channels,
            in_depth, in_height, in_width,
            kernel_d, kernel_h, kernel_w,
            out_depth, out_height, out_width,
            stride, padding, dilation, groups
        );
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid 3D convolution forward (CUDA)");
}