#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

__global__ void conv_transpose2d_forward_kernel_vectorized(
    const float4* __restrict__ input4,
    const float4* __restrict__ weight4,
    const float* __restrict__ bias,
    float4* __restrict__ output4,
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
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int out_w_base = (tid * 4) % out_width;
    const int out_h = ((tid * 4) / out_width) % out_height;
    const int bo_idx = blockIdx.y;
    
    if (out_w_base >= out_width - 3 || out_h >= out_height)
        return;
    
    const int o = bo_idx % out_channels;
    const int b = bo_idx / out_channels;
    
    const float bias_val = __ldg(&bias[o]);
    
    float4 result;
    result.x = bias_val;
    result.y = bias_val;
    result.z = bias_val;
    result.w = bias_val;
    
    for (int c = 0; c < in_channels; c++) {
        for (int p = 0; p < kernel_size; p++) {
            #pragma unroll
            for (int q = 0; q < kernel_size; q++) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int out_w = out_w_base + i;
                    if (out_w >= out_width) continue;
                    
                    const int h_unscaled = out_h + padding - p * dilation;
                    const int w_unscaled = out_w + padding - q * dilation;
                    
                    if (h_unscaled % stride == 0 && w_unscaled % stride == 0) {
                        const int h_in = h_unscaled / stride;
                        const int w_in = w_unscaled / stride;
                        
                        if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                            const int input_idx4 = ((b * in_channels + c) * in_height + h_in) * ((in_width + 3) / 4);
                            const int input_offset = w_in / 4;
                            const float4 in4 = __ldg(&input4[input_idx4 + input_offset]);
                            
                            const int weight_idx4 = ((c * out_channels + o) * kernel_size + p) * ((kernel_size + 3) / 4);
                            const int weight_offset = q / 4;
                            const float4 weight4_val = __ldg(&weight4[weight_idx4 + weight_offset]);
                            
                            float in_val;
                            float weight_val;
                            
                            switch(w_in % 4) {
                                case 0: in_val = in4.x; break;
                                case 1: in_val = in4.y; break;
                                case 2: in_val = in4.z; break;
                                case 3: in_val = in4.w; break;
                            }
                            
                            switch(q % 4) {
                                case 0: weight_val = weight4_val.x; break;
                                case 1: weight_val = weight4_val.y; break;
                                case 2: weight_val = weight4_val.z; break;
                                case 3: weight_val = weight4_val.w; break;
                            }
                            
                            switch(i) {
                                case 0: result.x += in_val * weight_val; break;
                                case 1: result.y += in_val * weight_val; break;
                                case 2: result.z += in_val * weight_val; break;
                                case 3: result.w += in_val * weight_val; break;
                            }
                        }
                    }
                }
            }
        }
    }
    
    const int output_idx4 = ((b * out_channels + o) * out_height + out_h) * ((out_width + 3) / 4) + (out_w_base / 4);
    output4[output_idx4] = result;
}

torch::Tensor conv_transpose2d_forward_cuda_vectorized(
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
    
    const int threads = 256;
    const int blocks_x = (out_height * ((out_width + 3) / 4) + threads - 1) / threads;
    const dim3 blocks(blocks_x, batch_size * out_channels);
    
    conv_transpose2d_forward_kernel_vectorized<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(input.data_ptr<float>()),
        reinterpret_cast<const float4*>(weight.data_ptr<float>()),
        bias.data_ptr<float>(),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
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

torch::Tensor conv_transpose2d_forward_wrapper_vectorized(
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
    
    return conv_transpose2d_forward_cuda_vectorized(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward_wrapper_vectorized,
          "ConvTranspose2d forward vectorized (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
}