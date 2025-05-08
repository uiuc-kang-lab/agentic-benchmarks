#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

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
    
    const int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    const int bo_idx = blockIdx.z;
    
    if (out_w >= out_width || out_h >= out_height)
        return;
    
    const int o = bo_idx % out_channels;
    const int b = bo_idx / out_channels;
    
    float result = __ldg(&bias[o]);
    
    const int TILE_SIZE = 4;
    #pragma unroll
    for (int c_base = 0; c_base < in_channels; c_base += TILE_SIZE) {
        float temp_results[TILE_SIZE] = {0.0f};
        
        #pragma unroll
        for (int p = 0; p < kernel_size; p++) {
            const int h_unscaled = out_h + padding - p * dilation;
            if (h_unscaled % stride != 0)
                continue;
                
            const int h_in = h_unscaled / stride;
            if (h_in < 0 || h_in >= in_height)
                continue;
                
            #pragma unroll
            for (int q = 0; q < kernel_size; q++) {
                const int w_unscaled = out_w + padding - q * dilation;
                if (w_unscaled % stride != 0)
                    continue;
                    
                const int w_in = w_unscaled / stride;
                if (w_in < 0 || w_in >= in_width)
                    continue;
                
                #pragma unroll
                for (int c_offset = 0; c_offset < TILE_SIZE && c_base + c_offset < in_channels; c_offset++) {
                    const int c = c_base + c_offset;
                    const int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
                    const int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
                    temp_results[c_offset] += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                }
            }
        }
        
        #pragma unroll
        for (int i = 0; i < TILE_SIZE && c_base + i < in_channels; i++) {
            result += temp_results[i];
        }
    }
    
    const int output_idx = ((b * out_channels + o) * out_height + out_h) * out_width + out_w;
    output[output_idx] = result;
}

torch::Tensor conv_transpose2d_forward_cuda_optimized(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);
    
    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    const dim3 threads(16, 16);
    const dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        batch_size * out_channels
    );
    
    conv_transpose2d_forward_kernel_optimized<<<blocks, threads>>>(
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

torch::Tensor conv_transpose2d_forward_wrapper_optimized(
    torch::Tensor input,
    torch::Tensor weight,
    pybind11::object bias_obj,
    int stride,
    int padding,
    int dilation) {
    
    const int out_channels = weight.size(1);
    torch::Tensor bias;
    if (bias_obj.is(pybind11::none())) {
        bias = torch::zeros({out_channels}, weight.options());
    } else {
        bias = bias_obj.cast<torch::Tensor>();
    }
    
    return conv_transpose2d_forward_cuda_optimized(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward_wrapper_optimized,
          "ConvTranspose2d forward optimized with __ldg (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
}
