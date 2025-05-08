#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define MAX_KERNEL_SIZE 11
#define MAX_CHANNELS 64

__constant__ float const_weight[MAX_CHANNELS * MAX_CHANNELS * MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_size,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding) {
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    const int x = bx * BLOCK_SIZE + tx;
    const int y = by * BLOCK_SIZE + ty;
    const int b = bz;
    
    if (x < output_width && y < output_height && b < batch_size) {
        for (int oc = 0; oc < out_channels; ++oc) {
            float sum = 0.0f;
            
            #pragma unroll 4
            for (int ic = 0; ic < in_channels; ++ic) {
                #pragma unroll
                for (int kh = 0; kh < kernel_size; ++kh) {
                    #pragma unroll
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        const int ih = y * stride - padding + kh;
                        const int iw = x * stride - padding + kw;
                        
                        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                            const int input_idx = ((b * in_channels + ic) * input_height + ih) * input_width + iw;
                            const int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                            sum += input[input_idx] * const_weight[weight_idx];
                        }
                    }
                }
            }
            
            const int output_idx = ((b * out_channels + oc) * output_height + y) * output_width + x;
            output[output_idx] = sum;
        }
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
    
    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto input_height = x.size(2);
    const auto input_width = x.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);
    
    TORCH_CHECK(kernel_size <= MAX_KERNEL_SIZE, "Kernel size exceeds maximum supported size");
    TORCH_CHECK(in_channels <= MAX_CHANNELS && out_channels <= MAX_CHANNELS, "Channel dimensions exceed maximum supported size");
    
    const auto output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    const auto output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width},
                             x.options());
    
    cudaMemcpyToSymbol(const_weight, 
                       weight.data_ptr<float>(), 
                       out_channels * in_channels * kernel_size * kernel_size * sizeof(float));
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((output_width + threads.x - 1) / threads.x,
                (output_height + threads.y - 1) / threads.y,
                batch_size);
    
    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        output_height,
        output_width,
        stride,
        padding);
    
    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward convolution with constant memory optimization");
}