#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

constexpr int WARP_SIZE = 32;
constexpr int ELEMENTS_PER_THREAD = 4;
constexpr int BLOCK_SIZE = 128;
constexpr int TILE_SIZE = 32;

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_tile_max(
    const scalar_t* __restrict__ input,
    const int b, const int c,
    const int oh, const int ow,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int input_height,
    const int input_width,
    const int input_stride_batch,
    const int input_stride_channel
) {
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    __shared__ scalar_t tile[TILE_SIZE][TILE_SIZE];
    
    const int tile_h_start = oh * stride - padding;
    const int tile_w_start = ow * stride - padding;
    
    for (int kh = threadIdx.x; kh < kernel_size; kh += BLOCK_SIZE) {
        const int ih = tile_h_start + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            for (int kw = 0; kw < kernel_size; kw++) {
                const int iw = tile_w_start + kw * dilation;
                if (iw >= 0 && iw < input_width) {
                    const int input_idx = b * input_stride_batch +
                                        c * input_stride_channel +
                                        ih * input_width + iw;
                    tile[kh][kw] = input[input_idx];
                }
            }
        }
    }
    __syncthreads();
    
    for (int kh = 0; kh < kernel_size; kh++) {
        const int ih = tile_h_start + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            for (int kw = 0; kw < kernel_size; kw++) {
                const int iw = tile_w_start + kw * dilation;
                if (iw >= 0 && iw < input_width) {
                    max_val = max(max_val, tile[kh][kw]);
                }
            }
        }
    }
    return max_val;
}

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
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    
    const int total_elements = batch_size * channels * output_height * output_width;
    const int elements_per_warp = WARP_SIZE * ELEMENTS_PER_THREAD;
    const int warp_offset = warp_id * elements_per_warp;
    
    const int input_stride_batch = channels * input_height * input_width;
    const int input_stride_channel = input_height * input_width;
    
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int element_idx = warp_offset + lane_id + i * WARP_SIZE;
        if (element_idx >= total_elements) return;
        
        const int w = element_idx % output_width;
        const int h = (element_idx / output_width) % output_height;
        const int c = (element_idx / (output_width * output_height)) % channels;
        const int b = element_idx / (output_width * output_height * channels);
        
        output[element_idx] = compute_tile_max<scalar_t>(
            input, b, c, h, w,
            kernel_size, stride, padding, dilation,
            input_height, input_width,
            input_stride_batch, input_stride_channel
        );
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
    const int elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int blocks = (total_elements + elements_per_block - 1) / elements_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, BLOCK_SIZE>>>(
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