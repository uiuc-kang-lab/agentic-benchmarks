#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024
#define TILE_SIZE 8

__global__ void conv3d_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    const int oc = blockIdx.x;
    const int batch_id = blockIdx.y;
    
    // Use thread ID to compute 3D position
    const int total_elements = out_depth * out_height * out_width;
    
    // Cache bias value for current output channel
    const float bias_val = (bias != nullptr) ? __ldg(&bias[oc]) : 0.0f;
    
    // Each thread processes multiple elements
    #pragma unroll 1
    for (int idx = tid; idx < total_elements; idx += block_size) {
        const int od = idx / (out_height * out_width);
        const int tmp = idx % (out_height * out_width);
        const int oh = tmp / out_width;
        const int ow = tmp % out_width;
        
        float sum = 0.0f;
        
        // Pre-compute base indices for input and weight
        const int batch_offset = batch_id * in_channels * in_depth * in_height * in_width;
        const int weight_base = oc * in_channels * kernel_d * kernel_h * kernel_w;
        
        #pragma unroll 2
        for (int ic = 0; ic < in_channels; ++ic) {
            const int input_channel_offset = batch_offset + ic * in_depth * in_height * in_width;
            const int weight_channel_offset = weight_base + ic * kernel_d * kernel_h * kernel_w;
            
            #pragma unroll 2
            for (int kd = 0; kd < kernel_d; ++kd) {
                const int id = od * stride - padding + kd * dilation;
                if (id >= 0 && id < in_depth) {
                    const int input_d_offset = input_channel_offset + id * in_height * in_width;
                    const int weight_d_offset = weight_channel_offset + kd * kernel_h * kernel_w;
                    
                    #pragma unroll 2
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        const int ih = oh * stride - padding + kh * dilation;
                        if (ih >= 0 && ih < in_height) {
                            const int input_h_offset = input_d_offset + ih * in_width;
                            const int weight_h_offset = weight_d_offset + kh * kernel_w;
                            
                            #pragma unroll 4
                            for (int kw = 0; kw < kernel_w; kw += 2) {
                                const int iw = ow * stride - padding + kw * dilation;
                                
                                if (iw >= 0 && iw < in_width - 1 && kw < kernel_w - 1) {
                                    // Load two input elements at once
                                    float2 input_val;
                                    input_val.x = __ldg(&input[input_h_offset + iw]);
                                    input_val.y = __ldg(&input[input_h_offset + iw + dilation]);
                                    
                                    // Load two weight elements at once
                                    float2 weight_val;
                                    weight_val.x = __ldg(&weight[weight_h_offset + kw]);
                                    weight_val.y = __ldg(&weight[weight_h_offset + kw + 1]);
                                    
                                    sum += input_val.x * weight_val.x + input_val.y * weight_val.y;
                                } else if (iw >= 0 && iw < in_width) {
                                    // Handle edge cases with single loads
                                    sum += __ldg(&input[input_h_offset + iw]) * 
                                          __ldg(&weight[weight_h_offset + kw]);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Write output with bias
        const int output_idx = ((batch_id * out_channels + oc) * out_depth + od) *
                               out_height * out_width + oh * out_width + ow;
        output[output_idx] = sum + bias_val;
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    TORCH_CHECK(groups == 1, "Only groups=1 is supported");
    auto bias = bias_opt.value_or(at::Tensor());
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    
    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width},
                           input.options());
    
    dim3 grid(out_channels, batch_size);
    int num_threads = BLOCK_SIZE;
    
    conv3d_optimized_kernel<<<grid, num_threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, in_depth, in_height, in_width,
        out_channels, kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, dilation
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 3D convolution forward with improved memory access (CUDA)");
}