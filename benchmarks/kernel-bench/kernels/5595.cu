#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t, int TILE_SIZE, int KERNEL_SIZE>
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
    extern __shared__ char shared_mem[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_mem);

    const int tile_w = TILE_SIZE + (KERNEL_SIZE - 1) * dilation;
    const int tile_h = TILE_SIZE + (KERNEL_SIZE - 1) * dilation;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * TILE_SIZE;
    const int by = blockIdx.y * TILE_SIZE;
    const int bz = blockIdx.z;
    
    const int batch_idx = bz / channels;
    const int channel_idx = bz % channels;
    
    const int input_batch_offset = batch_idx * channels * input_height * input_width;
    const int input_channel_offset = channel_idx * input_height * input_width;

    // Load input tile into shared memory
    #pragma unroll
    for (int i = ty; i < tile_h; i += blockDim.y) {
        #pragma unroll
        for (int j = tx; j < tile_w; j += blockDim.x) {
            const int ih = by + i - padding;
            const int iw = bx + j - padding;
            
            const int shared_idx = i * tile_w + j;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                shared_input[shared_idx] = __ldg(&input[input_batch_offset + input_channel_offset + ih * input_width + iw]);
            } else {
                shared_input[shared_idx] = -std::numeric_limits<scalar_t>::infinity();
            }
        }
    }
    
    __syncthreads();

    // Compute max pooling
    const int out_x = bx + tx;
    const int out_y = by + ty;
    
    if (out_x < output_width && out_y < output_height) {
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        if constexpr (KERNEL_SIZE == 2) {
            #pragma unroll
            for (int kh = 0; kh < 2; kh++) {
                #pragma unroll
                for (int kw = 0; kw < 2; kw++) {
                    const int sh_y = ty + kh * dilation;
                    const int sh_x = tx + kw * dilation;
                    max_val = max(max_val, shared_input[sh_y * tile_w + sh_x]);
                }
            }
        } else if constexpr (KERNEL_SIZE == 3) {
            #pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                #pragma unroll
                for (int kw = 0; kw < 3; kw++) {
                    const int sh_y = ty + kh * dilation;
                    const int sh_x = tx + kw * dilation;
                    max_val = max(max_val, shared_input[sh_y * tile_w + sh_x]);
                }
            }
        } else {
            for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                    const int sh_y = ty + kh * dilation;
                    const int sh_x = tx + kw * dilation;
                    max_val = max(max_val, shared_input[sh_y * tile_w + sh_x]);
                }
            }
        }

        if (out_x < output_width && out_y < output_height) {
            const int output_idx = batch_idx * (channels * output_height * output_width) +
                                 channel_idx * (output_height * output_width) +
                                 out_y * output_width +
                                 out_x;
            output[output_idx] = max_val;
        }
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

    constexpr int TILE_SIZE = 16;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (output_width + TILE_SIZE - 1) / TILE_SIZE,
        (output_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * channels
    );

    const int tile_w = TILE_SIZE + (kernel_size - 1) * dilation;
    const int tile_h = TILE_SIZE + (kernel_size - 1) * dilation;
    const int shared_mem_size = tile_w * tile_h * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            max_pool2d_shared_kernel<scalar_t, TILE_SIZE, 2><<<blocks, threads, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else if (kernel_size == 3) {
            max_pool2d_shared_kernel<scalar_t, TILE_SIZE, 3><<<blocks, threads, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else {
            max_pool2d_shared_kernel<scalar_t, TILE_SIZE, -1><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}