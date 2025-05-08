#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t, int KERNEL_SIZE=-1>
__global__ void max_pool2d_hybrid_kernel(
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
    const int dilation,
    const bool use_shared
) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;

    if (b >= batch_size || c >= channels || oh >= output_height || ow >= output_width) return;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    if (use_shared) {
        extern __shared__ char shared_mem[];
        scalar_t* tile = reinterpret_cast<scalar_t*>(shared_mem);
        
        const int sm_w = blockDim.x * stride + ((KERNEL_SIZE > 0 ? KERNEL_SIZE : kernel_size) - 1) * dilation;
        const int sm_h = blockDim.y * stride + ((KERNEL_SIZE > 0 ? KERNEL_SIZE : kernel_size) - 1) * dilation;
        
        const int in_tile_x = blockIdx.x * blockDim.x * stride - padding;
        const int in_tile_y = blockIdx.y * blockDim.y * stride - padding;
        
        for (int ty = threadIdx.y; ty < sm_h; ty += blockDim.y) {
            for (int tx = threadIdx.x; tx < sm_w; tx += blockDim.x) {
                const int in_x = in_tile_x + tx;
                const int in_y = in_tile_y + ty;
                scalar_t value;
                if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height) {
                    value = __ldg(&input[((b * channels + c) * input_height + in_y) * input_width + in_x]);
                } else {
                    value = -std::numeric_limits<scalar_t>::infinity();
                }
                tile[ty * sm_w + tx] = value;
            }
        }
        __syncthreads();

        const int k_size = KERNEL_SIZE > 0 ? KERNEL_SIZE : kernel_size;
        const int tile_x = threadIdx.x * stride;
        const int tile_y = threadIdx.y * stride;
        
        #pragma unroll
        for (int kh = 0; kh < k_size; kh++) {
            #pragma unroll
            for (int kw = 0; kw < k_size; kw++) {
                const int sx = tile_x + kw * dilation;
                const int sy = tile_y + kh * dilation;
                max_val = fmaxf(max_val, tile[sy * sm_w + sx]);
            }
        }
    } else {
        const int input_base = ((b * channels + c) * input_height) * input_width;
        
        #pragma unroll
        for (int kh = 0; kh < (KERNEL_SIZE > 0 ? KERNEL_SIZE : kernel_size); kh++) {
            const int ih = oh * stride - padding + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                #pragma unroll
                for (int kw = 0; kw < (KERNEL_SIZE > 0 ? KERNEL_SIZE : kernel_size); kw++) {
                    const int iw = ow * stride - padding + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        max_val = fmaxf(max_val, __ldg(&input[input_base + ih * input_width + iw]));
                    }
                }
            }
        }
    }

    output[((b * channels + c) * output_height + oh) * output_width + ow] = max_val;
}