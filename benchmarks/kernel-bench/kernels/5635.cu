#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__device__ __forceinline__ float4 load_input4(
    const scalar_t* __restrict__ input,
    const int idx,
    const int stride) {
    float4 data;
    data.x = static_cast<float>(input[idx]);
    data.y = static_cast<float>(input[idx + stride]);
    data.z = static_cast<float>(input[idx + 2 * stride]);
    data.w = static_cast<float>(input[idx + 3 * stride]);
    return data;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_max_value(
    const scalar_t* __restrict__ input,
    const int b, const int c,
    const int oh, const int ow,
    const int input_height, const int input_width,
    const int kernel_size, const int stride,
    const int padding, const int dilation,
    const int channel_stride) {
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    #pragma unroll
    for (int kh = 0; kh < kernel_size; ++kh) {
        const int ih = oh * stride - padding + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            const int ih_offset = ih * input_width;
            
            #pragma unroll
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int iw = ow * stride - padding + kw * dilation;
                if (iw >= 0 && iw < input_width) {
                    const int input_idx = b * channel_stride + c * input_height * input_width + ih_offset + iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    }
    return max_val;
}

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding,
    const int dilation) {
    
    const int channel_stride = channels * input_height * input_width;
    const int output_stride = channels * output_height * output_width;
    
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int elements_per_thread = 4;
    
    for (int base_idx = thread_idx * elements_per_thread; 
         base_idx < batch_size * channels * output_height * output_width; 
         base_idx += total_threads * elements_per_thread) {
        
        const int ow = base_idx % output_width;
        const int oh = (base_idx / output_width) % output_height;
        const int c = (base_idx / (output_width * output_height)) % channels;
        const int b = base_idx / (output_width * output_height * channels);
        
        if (ow + 3 < output_width) {
            scalar_t max_vals[4];
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                max_vals[i] = compute_max_value<scalar_t>(
                    input, b, c, oh, ow + i,
                    input_height, input_width,
                    KERNEL_SIZE, stride, padding, dilation,
                    channel_stride
                );
            }
            
            if (sizeof(scalar_t) == 4) {
                float4* out_ptr = reinterpret_cast<float4*>(&output[base_idx]);
                float4 result;
                result.x = max_vals[0];
                result.y = max_vals[1];
                result.z = max_vals[2];
                result.w = max_vals[3];
                *out_ptr = result;
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    output[base_idx + i] = max_vals[i];
                }
            }
        } else {
            for (int i = 0; i < 4 && (ow + i) < output_width; ++i) {
                output[base_idx + i] = compute_max_value<scalar_t>(
                    input, b, c, oh, ow + i,
                    input_height, input_width,
                    KERNEL_SIZE, stride, padding, dilation,
                    channel_stride
                );
            }
        }
    }
}

torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {
    
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);
    
    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    
    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());
    
    const int threads = 256;
    const int blocks = std::min(65535, (batch_size * channels * output_height * output_width + threads * 4 - 1) / (threads * 4));
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        switch(kernel_size) {
            case 2:
                max_pool2d_kernel<scalar_t, 2><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation);
                break;
            case 3:
                max_pool2d_kernel<scalar_t, 3><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation);
                break;
            default:
                max_pool2d_kernel<scalar_t, -1><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation);
        }
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}