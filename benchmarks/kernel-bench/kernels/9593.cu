#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define MAX_WEIGHT_SIZE 1024
#define MAX_BIAS_SIZE 256
#define TILE_SIZE 16

__constant__ float const_weights[MAX_WEIGHT_SIZE];
__constant__ float const_bias[MAX_BIAS_SIZE];

template <typename scalar_t>
__global__ void optimizedDepthwiseConv2DKernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding)
{
    __shared__ scalar_t shared_input[TILE_SIZE + 2][TILE_SIZE + 2];
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int c = bz % in_channels;
    const int n = bz / in_channels;
    
    const int h_out_start = by * TILE_SIZE;
    const int w_out_start = bx * TILE_SIZE;
    const int h_out = h_out_start + ty;
    const int w_out = w_out_start + tx;
    
    if (n >= batch_size) return;
    
    const int h_in_base = h_out * stride - padding;
    const int w_in_base = w_out * stride - padding;
    
    scalar_t sum = 0;
    
    if (h_out < out_height && w_out < out_width) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                const int h_in = h_in_base + kh;
                const int w_in = w_in_base + kw;
                
                if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                    const int x_idx = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                    const int w_idx = (c * kernel_size + kh) * kernel_size + kw;
                    sum += x[x_idx] * const_weights[w_idx];
                }
            }
        }
        
        sum += const_bias[c];
        const int out_idx = ((n * in_channels + c) * out_height + h_out) * out_width + w_out;
        out[out_idx] = sum;
    }
}

torch::Tensor forward_impl(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int groups)
{
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int kernel_size = weight.size(2);
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());
    
    cudaMemcpyToSymbol(const_weights, weight.data_ptr<float>(), 
                       in_channels * kernel_size * kernel_size * sizeof(float));
    cudaMemcpyToSymbol(const_bias, bias.data_ptr<float>(), 
                       in_channels * sizeof(float));
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * in_channels
    );
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "optimized_depthwise_conv2d", ([&] {
        optimizedDepthwiseConv2DKernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size, in_channels, in_height, in_width,
            kernel_size, out_height, out_width,
            stride, padding
        );
    }));
    
    return out;
}