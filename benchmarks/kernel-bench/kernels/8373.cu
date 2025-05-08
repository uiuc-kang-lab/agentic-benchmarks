#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_height,
    const int kernel_width,
    const int output_height,
    const int output_width,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w) {
    
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_ch = blockIdx.z;
    
    if (out_x >= output_width || out_y >= output_height || out_ch >= out_channels)
        return;
    
    for (int batch = 0; batch < batch_size; ++batch) {
        float sum = 0.0f;
        
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int in_x = (out_x + padding_w - kw) / stride_w;
                    int in_y = (out_y + padding_h - kh) / stride_h;
                    
                    if (in_x >= 0 && in_x < input_width &&
                        in_y >= 0 && in_y < input_height &&
                        (out_x + padding_w - kw) % stride_w == 0 &&
                        (out_y + padding_h - kh) % stride_h == 0) {
                        
                        float input_val = input[
                            batch * in_channels * input_height * input_width +
                            in_ch * input_height * input_width +
                            in_y * input_width + in_x];
                        
                        float weight_val = weight[
                            in_ch * out_channels * kernel_height * kernel_width +
                            out_ch * kernel_height * kernel_width +
                            kh * kernel_width + kw];
                        
                        sum += input_val * weight_val;
                    }
                }
            }
        }
        
        output[batch * out_channels * output_height * output_width +
               out_ch * output_height * output_width +
               out_y * output_width + out_x] = sum;
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
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_height = x.size(2);
    const int input_width = x.size(3);
    const int out_channels = weight.size(1);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);
    
    const int output_height = (input_height - 1) * stride[0] - 2 * padding[0] +
                              kernel_height + output_padding[0];
    const int output_width = (input_width - 1) * stride[1] - 2 * padding[1] +
                             kernel_width + output_padding[1];
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width},
                              x.options());
    
    dim3 threads(16, 16);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        out_channels
    );
    
    conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        output_height,
        output_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1]
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward (CUDA)");
}