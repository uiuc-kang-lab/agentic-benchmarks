#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16

__global__ void conv2d_kernel(
float* input,
float* weight,
float* output,
int batch_size,
int in_channels,
int out_channels,
int height,
int width,
int kernel_h,
int kernel_w,
int stride,
int padding) {

    __shared__ float shared_input[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int blockz = blockIdx.z;
    int oc = blockz % out_channels;
    int b = blockz / out_channels;
    
    int out_h = (height + 2 * padding - kernel_h) / stride + 1;
    int out_w = (width + 2 * padding - kernel_w) / stride + 1;
    
    int h = by * BLOCK_SIZE + ty;
    int w = bx * BLOCK_SIZE + tx;
    
    if (h < out_h && w < out_w) {
        float sum = 0.0f;
        
        for (int c = 0; c < in_channels; c++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int h_in = h * stride - padding + kh;
                    int w_in = w * stride - padding + kw;
                    
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        int input_idx = ((bz * in_channels + c) * height + h_in) * width + w_in;
                        int weight_idx = ((blockIdx.z * in_channels + c) * kernel_h + kh) * kernel_w + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        int out_idx = ((bz * out_channels + blockIdx.z) * out_h + h) * out_w + w;
        output[out_idx] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    
    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    auto out_h = (height + 2 * padding - kernel_h) / stride + 1;
    auto out_w = (width + 2 * padding - kernel_w) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w},
                              x.options());
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((out_w + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (out_h + BLOCK_SIZE - 1) / BLOCK_SIZE,
                batch_size * out_channels);
    
    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_h,
        kernel_w,
        stride,
        padding);
    
    if (bias.has_value()) {
        output += bias.value().view({1, -1, 1, 1});
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution");
}