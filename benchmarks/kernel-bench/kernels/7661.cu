#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define ITEMS_PER_THREAD 4

__global__ void conv3d_improved_strided_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size, const int in_channels, const int out_channels,
    const int in_depth, const int in_height, const int in_width,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int out_depth, const int out_height, const int out_width,
    const int stride, const int padding, const int dilation,
    const int groups, const int total_elements) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride_size = gridDim.x * blockDim.x;
    
    #pragma unroll 1
    for (int base_idx = tid * ITEMS_PER_THREAD; base_idx < total_elements; base_idx += stride_size * ITEMS_PER_THREAD) {
        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD && base_idx + item * stride_size < total_elements; ++item) {
            const int idx = base_idx + item * stride_size;
            
            const int w_out = idx % out_width;
            int temp = idx / out_width;
            const int h_out = temp % out_height;
            temp /= out_height;
            const int d_out = temp % out_depth;
            temp /= out_depth;
            const int c_out = temp % out_channels;
            const int b = temp / out_channels;
            
            const int group = c_out / (out_channels / groups);
            const int channels_per_group = in_channels / groups;
            
            float sum = 0.0f;
            
            #pragma unroll 2
            for (int ic = 0; ic < channels_per_group; ++ic) {
                const int in_c = group * channels_per_group + ic;
                
                for (int kd = 0; kd < kernel_d; ++kd) {
                    const int d_in = d_out * stride - padding + kd * dilation;
                    if (d_in >= 0 && d_in < in_depth) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            const int h_in = h_out * stride - padding + kh * dilation;
                            if (h_in >= 0 && h_in < in_height) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    const int w_in = w_out * stride - padding + kw * dilation;
                                    if (w_in >= 0 && w_in < in_width) {
                                        sum += input[((b * in_channels + in_c) * in_depth + d_in) * 
                                                     in_height * in_width + h_in * in_width + w_in] *
                                               weight[((c_out * channels_per_group + ic) * kernel_d + kd) * 
                                                      kernel_h * kernel_w + kh * kernel_w + kw];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            if (bias != nullptr) {
                sum += bias[c_out];
            }
            
            output[idx] = sum;
        }
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    auto bias = bias_opt.value_or(at::Tensor());
    
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(!bias.defined() || bias.is_cuda(), "Bias must be a CUDA tensor");
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);
    
    const int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    const int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    
    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int thread_count = BLOCK_SIZE;
    const int num_blocks = (total_elements + thread_count * ITEMS_PER_THREAD - 1) / (thread_count * ITEMS_PER_THREAD);
    
    conv3d_improved_strided_kernel<<<num_blocks, thread_count>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, dilation,
        groups, total_elements
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward with improved strided loops (CUDA)");
}