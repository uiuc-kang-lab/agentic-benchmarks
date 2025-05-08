#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_height,
    const int kernel_width,
    const int out_height,
    const int out_width,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w) {
    
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (out_x < out_width && out_y < out_height && b < batch_size) {
        for (int oc = 0; oc < out_channels; ++oc) {
            float sum = 0.0f;
            
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kh = 0; kh < kernel_height; ++kh) {
                    for (int kw = 0; kw < kernel_width; ++kw) {
                        const int in_y = (out_y + padding_h - kh) / stride_h;
                        const int in_x = (out_x + padding_w - kw) / stride_w;
                        
                        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width &&
                            (out_y + padding_h - kh) % stride_h == 0 &&
                            (out_x + padding_w - kw) % stride_w == 0) {
                            
                            const int input_idx = ((b * in_channels + ic) * in_height + in_y) * in_width + in_x;
                            const int weight_idx = ((oc * in_channels + ic) * kernel_height + kh) * kernel_width + kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            
            const int output_idx = ((b * out_channels + oc) * out_height + out_y) * out_width + out_x;
            output[output_idx] = sum;
        }
    }
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
    
    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto in_height = x.size(2);
    const auto in_width = x.size(3);
    const auto out_channels = weight.size(1);
    const auto kernel_height = weight.size(2);
    const auto kernel_width = weight.size(3);
    
    const auto out_height = (in_height - 1) * stride[0] - 2 * padding[0] + 
                            dilation[0] * (kernel_height - 1) + output_padding[0] + 1;
    const auto out_width = (in_width - 1) * stride[1] - 2 * padding[1] + 
                           dilation[1] * (kernel_width - 1) + output_padding[1] + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width},
                              x.options());
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        batch_size
    );
    
    conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_height,
        kernel_width,
        out_height,
        out_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1]
    );
    
    if (bias.has_value()) {
        output.add_(bias.value().view({1, out_channels, 1, 1}));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward (CUDA)");
}