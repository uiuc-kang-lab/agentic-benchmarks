#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

__global__ void conv_transpose2d_forward_kernel_uniform(
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
    
    // Align thread indexing to warp size (32 threads)
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    
    const int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    const int bo_idx = blockIdx.z;
    
    // Early exit for all threads in warp if any are out of bounds
    if ((out_w | out_h) >= max(out_width, out_height))
        return;
        
    const int o = bo_idx % out_channels;
    const int b = bo_idx / out_channels;
    
    float result = __ldg(&bias[o]);
    
    // Pre-compute stride-related values for the entire warp
    const int h_base = out_h + padding;
    const int w_base = out_w + padding;
    
    // Process channels in vectors of 4 for better memory coalescing
    const int vec_size = 4;
    const int in_channels_aligned = (in_channels + vec_size - 1) & ~(vec_size - 1);
    
    for (int c_base = 0; c_base < in_channels_aligned; c_base += vec_size) {
        float4 temp_result = {0.0f, 0.0f, 0.0f, 0.0f};
        
        #pragma unroll
        for (int p = 0; p < kernel_size; p++) {
            const int h_offset = h_base - p * dilation;
            const bool h_valid = (h_offset >= 0) && (h_offset < out_height * stride);
            const int h_in = h_offset / stride;
            
            // Skip entire row if not valid for any thread in warp
            if (!__any_sync(0xffffffff, h_valid && (h_offset % stride == 0)))
                continue;
                
            #pragma unroll
            for (int q = 0; q < kernel_size; q++) {
                const int w_offset = w_base - q * dilation;
                const bool w_valid = (w_offset >= 0) && (w_offset < out_width * stride);
                const int w_in = w_offset / stride;
                
                // Skip column if not valid for any thread in warp
                if (!__any_sync(0xffffffff, w_valid && (w_offset % stride == 0)))
                    continue;
                
                // Use predication instead of branching
                const bool valid = h_valid && w_valid && 
                                 (h_offset % stride == 0) && (w_offset % stride == 0) &&
                                 (h_in < in_height) && (w_in >= 0) && (w_in < in_width);
                
                #pragma unroll
                for (int c_offset = 0; c_offset < vec_size && c_base + c_offset < in_channels; c_offset++) {
                    const int c = c_base + c_offset;
                    const int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
                    const int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
                    
                    const float input_val = valid ? __ldg(&input[input_idx]) : 0.0f;
                    const float weight_val = __ldg(&weight[weight_idx]);
                    
                    reinterpret_cast<float*>(&temp_result)[c_offset] += input_val * weight_val;
                }
            }
        }
        
        result += temp_result.x + temp_result.y + temp_result.z + temp_result.w;
    }
    
    if (out_w < out_width && out_h < out_height) {
        const int output_idx = ((b * out_channels + o) * out_height + out_h) * out_width + out_w;
        output[output_idx] = result;
    }
}

torch::Tensor conv_transpose2d_forward_cuda_uniform(
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
    
    const dim3 threads(32, 4);
    const dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        batch_size * out_channels
    );
    
    conv_transpose2d_forward_kernel_uniform<<<blocks, threads>>>(
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

torch::Tensor conv_transpose2d_forward_wrapper_uniform(
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
    
    return conv_transpose2d_forward_cuda_uniform(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward_wrapper_uniform,
          "ConvTranspose2d forward with uniform warp execution (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
}