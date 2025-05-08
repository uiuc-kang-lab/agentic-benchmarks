#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_hybrid_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding,
    const int dilation
) {
    __shared__ scalar_t shared_input[TILE_SIZE][TILE_SIZE];
    
    const int output_elements = output_height * output_width;
    const int total_elements = batch_channels * output_elements;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_elements; 
         idx += gridDim.x * blockDim.x) {
        
        const int bc = idx / output_elements;
        const int oh = (idx % output_elements) / output_width;
        const int ow = idx % output_width;

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        const int tile_row = threadIdx.y;
        const int tile_col = threadIdx.x;
        if (tile_row < TILE_SIZE && tile_col < TILE_SIZE) {
            const int ih = blockIdx.y * TILE_SIZE + tile_row;
            const int iw = blockIdx.x * TILE_SIZE + tile_col;
            if (ih < input_height && iw < input_width) {
                shared_input[tile_row][tile_col] = input[(bc * input_height + ih) * input_width + iw];
            }
        }
        __syncthreads();

        if (KERNEL_SIZE == 2) {
            #pragma unroll
            for (int kh = 0; kh < 2; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < 2; ++kw) {
                    const int ih = oh * stride - padding + kh * dilation;
                    const int iw = ow * stride - padding + kw * dilation;
                    
                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                        if (ih / TILE_SIZE == blockIdx.y && iw / TILE_SIZE == blockIdx.x) {
                            max_val = max(max_val, shared_input[ih % TILE_SIZE][iw % TILE_SIZE]);
                        } else {
                            max_val = max(max_val, input[(bc * input_height + ih) * input_width + iw]);
                        }
                    }
                }
            }
        } else {
            for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
                for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                    const int ih = oh * stride - padding + kh * dilation;
                    const int iw = ow * stride - padding + kw * dilation;
                    
                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                        if (ih / TILE_SIZE == blockIdx.y && iw / TILE_SIZE == blockIdx.x) {
                            max_val = max(max_val, shared_input[ih % TILE_SIZE][iw % TILE_SIZE]);
                        } else {
                            max_val = max(max_val, input[(bc * input_height + ih) * input_width + iw]);
                        }
                    }
                }
            }
        }
        
        output[idx] = max_val;
    }
}

torch::Tensor max_pool2d_cuda_forward_hybrid(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    
    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());
    
    const int batch_channels = batch_size * channels;
    const dim3 threads(16, 16);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_forward_hybrid", ([&] {
        if (kernel_size == 2) {
            max_pool2d_hybrid_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_channels,
                input_height,
                input_width,
                output_height,
                output_width,
                stride,
                padding,
                dilation
            );
        } else {
            max_pool2d_hybrid_kernel<scalar_t, -1><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_channels,
                input_height,
                input_width,
                output_height,
                output_width,
                stride,
                padding,
                dilation
            );
        }
    }));

    return output;
}