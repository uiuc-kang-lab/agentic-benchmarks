#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

constexpr int WARP_SIZE = 32;
constexpr int ELEMENTS_PER_THREAD = 4;
constexpr int TILE_SIZE = 4;  // Tile size for shared memory optimization

template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    __shared__ scalar_t shared_input[TILE_SIZE][(TILE_SIZE + 2) * (TILE_SIZE + 2)];
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    
    const int total_elements = batch_size * channels * output_height * output_width;
    const int elements_per_warp = WARP_SIZE * ELEMENTS_PER_THREAD;
    const int warp_offset = warp_id * elements_per_warp;
    
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int element_idx = warp_offset + lane_id + i * WARP_SIZE;
        if (element_idx >= total_elements) return;
        
        const int w = element_idx % output_width;
        const int h = (element_idx / output_width) % output_height;
        const int c = (element_idx / (output_width * output_height)) % channels;
        const int b = element_idx / (output_width * output_height * channels);
        
        const int tile_start_h = (h * stride - padding) / TILE_SIZE * TILE_SIZE;
        const int tile_start_w = (w * stride - padding) / TILE_SIZE * TILE_SIZE;
        
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        const int input_batch_offset = b * channels * input_height * input_width;
        const int input_channel_offset = c * input_height * input_width;
        
        if (lane_id < TILE_SIZE * TILE_SIZE) {
            const int tile_h = lane_id / TILE_SIZE;
            const int tile_w = lane_id % TILE_SIZE;
            const int ih = tile_start_h + tile_h;
            const int iw = tile_start_w + tile_w;
            
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                const int input_idx = input_batch_offset + input_channel_offset + ih * input_width + iw;
                shared_input[tile_h][tile_w] = input[input_idx];
            }
        }
        __syncwarp();
        
        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            const int ih = h * stride - padding + kh * dilation - tile_start_h;
            if (ih >= 0 && ih < TILE_SIZE) {
                #pragma unroll
                for (int kw = 0; kw < kernel_size; kw++) {
                    const int iw = w * stride - padding + kw * dilation - tile_start_w;
                    if (iw >= 0 && iw < TILE_SIZE) {
                        max_val = max(max_val, shared_input[ih][iw]);
                    }
                }
            }
        }
        
        output[element_idx] = max_val;
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

    const int total_elements = batch_size * channels * output_height * output_width;
    const int threads_per_block = 128;
    const int elements_per_block = threads_per_block * ELEMENTS_PER_THREAD;
    const int blocks = (total_elements + elements_per_block - 1) / elements_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }));

    return output;
}