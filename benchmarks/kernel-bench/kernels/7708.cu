#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <ATen/cudnn/Handles.h>

#define BLOCK_SIZE 1024
#define SMALL_TENSOR_THRESHOLD 64 // Threshold for small tensor dimensions

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    auto bias = bias_opt.value_or(at::Tensor());
    
    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width},
                           input.options());

    // Choose implementation based on tensor size
    bool use_cudnn = (in_depth >= SMALL_TENSOR_THRESHOLD && 
                     in_height >= SMALL_TENSOR_THRESHOLD &&
                     in_width >= SMALL_TENSOR_THRESHOLD);

    if (use_cudnn) {
        // Use cuDNN implementation for large tensors
        cudnnHandle_t handle = at::native::getCudnnHandle();
        // ... (cuDNN implementation from Kernel 2)
        // Set up descriptors and run cuDNN convolution
        return cudnn_forward(handle, input, weight, bias, 
                           stride, padding, dilation, groups);
    } else {
        // Use custom CUDA kernel for small tensors
        dim3 grid(out_channels, batch_size);
        int num_threads = BLOCK_SIZE;
        
        conv3d_optimized_kernel<<<grid, num_threads>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_size, in_channels, in_depth, in_height, in_width,
            out_channels, kernel_d, kernel_h, kernel_w,
            out_depth, out_height, out_width,
            stride, padding, dilation
        );
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid 3D convolution forward (CUDA)");
}