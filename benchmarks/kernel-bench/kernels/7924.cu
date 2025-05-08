#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define KERNEL_SIZE 3
#define FEATURES_PER_BLOCK 4
#define TILE_SIZE (BLOCK_SIZE + KERNEL_SIZE - 1)

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
    
    __shared__ float shared_input[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_weights[FEATURES_PER_BLOCK][KERNEL_SIZE][KERNEL_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * BLOCK_SIZE;
    const int by = blockIdx.y * BLOCK_SIZE;
    const int b = blockIdx.z;
    
    const int x_out = bx + tx;
    const int y_out = by + ty;
    
    float partial_sums[FEATURES_PER_BLOCK] = {0.0f};
    
    for (int ocb = 0; ocb < out_channels; ocb += FEATURES_PER_BLOCK) {
        for (int ic = 0; ic < in_channels; ++ic) {
            if (tx < KERNEL_SIZE && ty < KERNEL_SIZE) {
                for (int f = 0; f < FEATURES_PER_BLOCK && (ocb + f) < out_channels; ++f) {
                    shared_weights[f][ty][tx] = weight[((ocb + f) * in_channels + ic) * KERNEL_SIZE * KERNEL_SIZE + ty * KERNEL_SIZE + tx];
                }
            }
            
            for (int i = ty; i < TILE_SIZE; i += BLOCK_SIZE) {
                for (int j = tx; j < TILE_SIZE; j += BLOCK_SIZE) {
                    int ih = by + i - padding;
                    int iw = bx + j - padding;
                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                        shared_input[i][j] = input[((b * in_channels + ic) * input_height + ih) * input_width + iw];
                    } else {
                        shared_input[i][j] = 0.0f;
                    }
                }
            }
            __syncthreads();
            
            if (x_out < output_width && y_out < output_height) {
                for (int f = 0; f < FEATURES_PER_BLOCK && (ocb + f) < out_channels; ++f) {
                    float sum = 0.0f;
                    #pragma unroll
                    for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
                        #pragma unroll
                        for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                            sum += shared_input[ty * stride + ky][tx * stride + kx] * 
                                   shared_weights[f][ky][kx];
                        }
                    }
                    partial_sums[f] += sum;
                }
            }
            __syncthreads();
        }
        
        if (x_out < output_width && y_out < output_height) {
            for (int f = 0; f < FEATURES_PER_BLOCK && (ocb + f) < out_channels; ++f) {
                output[((b * out_channels + (ocb + f)) * output_height + y_out) * output_width + x_out] = 
                    partial_sums[f];
                partial_sums[f] = 0.0f;
            }
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
    m.def("forward", &forward, "Shared memory optimized CUDA conv2d implementation");
}