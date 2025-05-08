#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_shared_kernel(
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
    constexpr int TILE_DIM_H = 16;
    constexpr int TILE_DIM_W = 16;
    constexpr int TILE_PAD = 2;
    
    __shared__ scalar_t shared_input[TILE_DIM_H + 2*TILE_PAD][TILE_DIM_W + 2*TILE_PAD];
    
    const int ow_base = blockIdx.x * TILE_DIM_W;
    const int oh_base = blockIdx.y * TILE_DIM_H;
    const int bc = blockIdx.z;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int b = bc / channels;
    const int c = bc % channels;
    
    const int input_b_offset = b * channels * input_height * input_width;
    const int input_c_offset = c * input_height * input_width;
    
    #pragma unroll
    for (int i = 0; i < ((TILE_DIM_H + 2*TILE_PAD + 15)/16); i++) {
        for (int j = 0; j < ((TILE_DIM_W + 2*TILE_PAD + 15)/16); j++) {
            const int sh = i * 16 + ty;
            const int sw = j * 16 + tx;
            if (sh < TILE_DIM_H + 2*TILE_PAD && sw < TILE_DIM_W + 2*TILE_PAD) {
                const int ih = oh_base + sh - TILE_PAD;
                const int iw = ow_base + sw - TILE_PAD;
                
                scalar_t val = -std::numeric_limits<scalar_t>::infinity();
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    val = __ldg(&input[input_b_offset + input_c_offset + ih * input_width + iw]);
                }
                shared_input[sh][sw] = val;
            }
        }
    }
    
    __syncthreads();
    
    const int ow = ow_base + tx;
    const int oh = oh_base + ty;
    
    if (ow < output_width && oh < output_height) {
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        if constexpr (KERNEL_SIZE == 2) {
            #pragma unroll
            for (int kh = 0; kh < 2; kh++) {
                #pragma unroll
                for (int kw = 0; kw < 2; kw++) {
                    const int sh = ty + kh * dilation + TILE_PAD;
                    const int sw = tx + kw * dilation + TILE_PAD;
                    max_val = max(max_val, shared_input[sh][sw]);
                }
            }
        } else if constexpr (KERNEL_SIZE == 3) {
            #pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                #pragma unroll
                for (int kw = 0; kw < 3; kw++) {
                    const int sh = ty + kh * dilation + TILE_PAD;
                    const int sw = tx + kw * dilation + TILE_PAD;
                    max_val = max(max_val, shared_input[sh][sw]);
                }
            }
        } else {
            for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                    const int sh = ty + kh * dilation + TILE_PAD;
                    const int sw = tx + kw * dilation + TILE_PAD;
                    max_val = max(max_val, shared_input[sh][sw]);
                }
            }
        }
        
        output[bc * output_height * output_width + oh * output_width + ow] = max_val;
    }
}

torch::Tensor max_pool2d_cuda_forward(
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
    
    const dim3 threads(16, 16);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            max_pool2d_shared_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else if (kernel_size == 3) {
            max_pool2d_shared_kernel<scalar_t, 3><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else {
            max_pool2d_shared_kernel<scalar_t, -1><<<blocks, threads>>>(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with shared memory (CUDA)");
}