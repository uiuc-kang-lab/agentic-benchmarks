#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <cuda_runtime.h>

namespace py = pybind11;

// Declare constant memory for weights (limited to 64KB on most GPUs)
__constant__ float d_weight[16384]; // Adjust size based on maximum expected weight size

__global__ void conv_transpose2d_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_height,
    const int kernel_width,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int out_height,
    const int out_width
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.z * blockDim.z + threadIdx.z;
    const int c = threadIdx.x;

    if (h < out_height && w < out_width && c < out_channels) {
        float sum = 0.0f;
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int in_h = (h + pad_h - kh) / stride_h;
                    int in_w = (w + pad_w - kw) / stride_w;
                    
                    if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                        if ((h + pad_h - kh) % stride_h == 0 && (w + pad_w - kw) % stride_w == 0) {
                            int in_idx = ((b * in_channels + ic) * in_height + in_h) * in_width + in_w;
                            int weight_idx = ((c * in_channels + ic) * kernel_height + kh) * kernel_width + kw;
                            sum += input[in_idx] * d_weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        int out_idx = ((b * out_channels + c) * out_height + h) * out_width + w;
        output[out_idx] = sum;
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    // Get dimensions
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(1);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);
    
    // Calculate output dimensions
    const int out_height = (in_height - 1) * stride[0] - 2 * padding[0] + kernel_height;
    const int out_width = (in_width - 1) * stride[1] - 2 * padding[1] + kernel_width;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());
    
    // Copy weight to constant memory
    cudaMemcpyToSymbol(d_weight, weight.data_ptr<float>(), 
                       weight.numel() * sizeof(float));
    
    // Launch kernel
    dim3 block(32, 8, 8);
    dim3 grid(batch_size,
              (out_height + block.y - 1) / block.y,
              (out_width + block.z - 1) / block.z);
    
    conv_transpose2d_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        out_height,
        out_width
    );
    
    // Add bias if present
    if (!bias_obj.is_none()) {
        auto bias = bias_obj.cast<torch::Tensor>();
        output.add_(bias.view({1, out_channels, 1, 1}));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}