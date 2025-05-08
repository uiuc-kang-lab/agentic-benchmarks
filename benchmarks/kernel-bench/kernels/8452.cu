#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>

#define MAX_WEIGHT_SIZE 8192  // Define maximum size for constant memory
__constant__ float d_weight[MAX_WEIGHT_SIZE];

__global__ void conv_transpose2d_kernel(
float* output,
const float* input,
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
const int out_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size * out_channels * out_height * out_width) return;
    
    int w_idx = idx % out_width;
    int h_idx = (idx / out_width) % out_height;
    int c_idx = (idx / (out_width * out_height)) % out_channels;
    int b_idx = idx / (out_width * out_height * out_channels);
    
    float sum = 0.0f;
    
    for(int ic = 0; ic < in_channels; ic++) {
        for(int kh = 0; kh < kernel_height; kh++) {
            for(int kw = 0; kw < kernel_width; kw++) {
                int in_h = (h_idx + pad_h - kh) / stride_h;
                int in_w = (w_idx + pad_w - kw) / stride_w;
                
                if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                    int input_idx = ((b_idx * in_channels + ic) * in_height + in_h) * in_width + in_w;
                    int weight_idx = ((c_idx * in_channels + ic) * kernel_height + kh) * kernel_width + kw;
                    sum += input[input_idx] * d_weight[weight_idx];
                }
            }
        }
    }
    
    output[idx] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {
    
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(x.dtype());
    
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int out_channels = weight.size(1) * groups;
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);
    
    int out_height = (in_height - 1) * stride[0] - 2 * padding[0] + kernel_height + output_padding[0];
    int out_width = (in_width - 1) * stride[1] - 2 * padding[1] + kernel_width + output_padding[1];
    
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, options);
    
    // Copy weight data to constant memory
    cudaMemcpyToSymbol(d_weight, weight.data_ptr<float>(), 
                       weight.numel() * sizeof(float));
    
    const int threads = 256;
    const int blocks = (batch_size * out_channels * out_height * out_width + threads - 1) / threads;
    
    conv_transpose2d_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_height, in_width,
        kernel_height, kernel_width,
        stride[0], stride[1],
        padding[0], padding[1],
        out_height, out_width);
    
    if (bias.has_value()) {
        output.add_(bias.value().view({1, out_channels, 1, 1}));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward (CUDA)");
}