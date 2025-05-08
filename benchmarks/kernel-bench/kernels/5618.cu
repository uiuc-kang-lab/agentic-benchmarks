#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void optimized_max_pool2d_kernel(
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
    const int dilation
) {
    extern __shared__ char shared_memory[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_memory);
    
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = blockIdx.z;
    
    if (ow >= output_width || oh >= output_height) return;

    const int b = bc / channels;
    const int c = bc % channels;

    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;
    
    const int input_batch_stride = channels * input_height * input_width;
    const int input_channel_stride = input_height * input_width;
    const int base_idx = b * input_batch_stride + c * input_channel_stride;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    if constexpr (KERNEL_SIZE == 2 || KERNEL_SIZE == 3) {
        const int K = KERNEL_SIZE;
        #pragma unroll
        for (int kh = 0; kh < K; ++kh) {
            const int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                const int row_idx = base_idx + ih * input_width;
                #pragma unroll
                for (int kw = 0; kw < K; ++kw) {
                    const int iw = iw_start + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        max_val = max(max_val, __ldg(&input[row_idx + iw]));
                    }
                }
            }
        }
    } else {
        const int tile_size = blockDim.x;
        const int input_tile_width = (KERNEL_SIZE - 1) * dilation + tile_size;
        
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            const int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                const int row_idx = base_idx + ih * input_width;
                
                for (int i = threadIdx.x; i < input_tile_width; i += blockDim.x) {
                    const int iw = iw_start + i;
                    if (iw >= 0 && iw < input_width) {
                        shared_input[threadIdx.y * input_tile_width + i] = 
                            input[row_idx + iw];
                    }
                }
                __syncthreads();
                
                #pragma unroll 4
                for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                    const int iw = kw * dilation + threadIdx.x;
                    if (iw < input_tile_width) {
                        max_val = max(max_val, 
                            shared_input[threadIdx.y * input_tile_width + iw]);
                    }
                }
                __syncthreads();
            }
        }
    }

    const int out_idx = b * (channels * output_height * output_width) + 
                       c * (output_height * output_width) + 
                       oh * output_width + ow;
    output[out_idx] = max_val;
}

torch::Tensor optimized_max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    dim3 threads(16, 16);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    const int shared_memory_size = threads.y * 
        ((kernel_size - 1) * dilation + threads.x) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            optimized_max_pool2d_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else if (kernel_size == 3) {
            optimized_max_pool2d_kernel<scalar_t, 3><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else {
            optimized_max_pool2d_kernel<scalar_t, -1><<<blocks, threads, shared_memory_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        }
    }));

    return output;
}