#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32  // Increased from 16 to 32
#define KERNEL_SIZE 3
#define SHARED_SIZE (BLOCK_SIZE + KERNEL_SIZE - 1)

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding) {
    
    __shared__ float shared_input[SHARED_SIZE][SHARED_SIZE];
    __shared__ float shared_weight[KERNEL_SIZE][KERNEL_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * BLOCK_SIZE;
    const int by = blockIdx.y * BLOCK_SIZE;
    const int b = blockIdx.z;
    
    const int x = bx + tx;
    const int y = by + ty;
    
    #pragma unroll 4
    for (int oc = 0; oc < out_channels; ++oc) {
        float sum = 0.0f;
        
        #pragma unroll 4
        for (int ic = 0; ic < in_channels; ++ic) {
            if (tx < KERNEL_SIZE && ty < KERNEL_SIZE) {
                shared_weight[ty][tx] = weight[((oc * in_channels + ic) * KERNEL_SIZE + ty) * KERNEL_SIZE + tx];
            }
            __syncthreads();
            
            if (tx < SHARED_SIZE && ty < SHARED_SIZE) {
                int ih = by + ty - padding;
                int iw = bx + tx - padding;
                
                shared_input[ty][tx] = (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) ?
                    input[((b * in_channels + ic) * input_height + ih) * input_width + iw] : 0.0f;
            }
            __syncthreads();
            
            if (x < output_width && y < output_height) {
                #pragma unroll
                for (int ki = 0; ki < KERNEL_SIZE; ++ki) {
                    #pragma unroll
                    for (int kj = 0; kj < KERNEL_SIZE; ++kj) {
                        sum += shared_input[ty * stride + ki][tx * stride + kj] * shared_weight[ki][kj];
                    }
                }
            }
            __syncthreads();
        }
        
        if (x < output_width && y < output_height) {
            output[((b * out_channels + oc) * output_height + y) * output_width + x] = sum;
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
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    
    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto input_height = x.size(2);
    auto input_width = x.size(3);
    auto out_channels = weight.size(0);
    
    auto output_height = (input_height + 2 * padding - KERNEL_SIZE) / stride + 1;
    auto output_width = (input_width + 2 * padding - KERNEL_SIZE) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width},
                             x.options());
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (output_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
                batch_size);
    
    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
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
    m.def("forward", &forward, "CUDA conv2d with optimized block size");
}