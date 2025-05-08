#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

__device__ inline void decode_index(int index, int out_width, int out_height, int out_channels,
                                    int &w_out, int &h_out, int &o, int &b) {
    w_out = index % out_width;
    int temp = index / out_width;
    h_out = temp % out_height;
    temp /= out_height;
    o = temp % out_channels;
    b = temp / out_channels;
}

__global__ void conv_transpose2d_forward_kernel_optimized(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation) {

    extern __shared__ float shared_weight[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (gid >= total_elements) return;
    
    int w_out, h_out, o, b;
    decode_index(gid, out_width, out_height, out_channels, w_out, h_out, o, b);
    
    int weight_elements = in_channels * kernel_size * kernel_size;
    for (int i = tid; i < weight_elements; i += blockDim.x) {
        int c = i / (kernel_size * kernel_size);
        int remaining = i % (kernel_size * kernel_size);
        int p = remaining / kernel_size;
        int q = remaining % kernel_size;
        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
        shared_weight[i] = weight[weight_idx];
    }
    __syncthreads();
    
    float out_val = bias[o];
    
    for (int c = 0; c < in_channels; ++c) {
        for (int p = 0; p < kernel_size; ++p) {
            int h_unscaled = h_out + padding - p * dilation;
            if (h_unscaled % stride != 0) continue;
            
            int h_in = h_unscaled / stride;
            if (h_in < 0 || h_in >= in_height) continue;
            
            for (int q = 0; q < kernel_size; ++q) {
                int w_unscaled = w_out + padding - q * dilation;
                if (w_unscaled % stride != 0) continue;
                
                int w_in = w_unscaled / stride;
                if (w_in < 0 || w_in >= in_width) continue;
                
                int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
                int weight_shared_idx = c * kernel_size * kernel_size + p * kernel_size + q;
                out_val += input[input_idx] * shared_weight[weight_shared_idx];
            }
        }
    }
    
    output[gid] = out_val;
}

torch::Tensor conv_transpose2d_forward_cuda_optimized(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    int total_elements = batch_size * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    size_t shared_mem_size = in_channels * kernel_size * kernel_size * sizeof(float);
    
    conv_transpose2d_forward_kernel_optimized<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_size,
        out_height,
        out_width,
        stride,
        padding,
        dilation);
        
    return output;
}