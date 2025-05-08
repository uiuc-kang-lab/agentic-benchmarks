#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

constexpr int WARP_SIZE = 32;
constexpr int ELEMENTS_PER_THREAD = 4;
constexpr int BLOCK_SIZE = 128;
constexpr int SHARED_MEM_SIZE = 32;

template <typename scalar_t>
__device__ __forceinline__ void calculate_indices(
    const int element_idx,
    const int output_width,
    const int output_height,
    const int channels,
    int& w,
    int& h,
    int& c,
    int& b
) {
    w = element_idx % output_width;
    h = (element_idx / output_width) % output_height;
    c = (element_idx / (output_width * output_height)) % channels;
    b = element_idx / (output_width * output_height * channels);
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
    __shared__ scalar_t shared_input[SHARED_MEM_SIZE][SHARED_MEM_SIZE];
    
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
        
        int w, h, c, b;
        calculate_indices<scalar_t>(
            element_idx, output_width, output_height, channels,
            w, h, c, b
        );
        
        const int input_stride_batch = channels * input_height * input_width;
        const int input_stride_channel = input_height * input_width;
        const int input_stride_height = input_width;
        
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        if (kernel_size <= SHARED_MEM_SIZE) {
            const int ih_start = h * stride - padding;
            const int iw_start = w * stride - padding;
            
            for (int sh = threadIdx.x; sh < kernel_size; sh += blockDim.x) {
                for (int sw = threadIdx.x; sw < kernel_size; sw += blockDim.x) {
                    const int ih = ih_start + sh * dilation;
                    const int iw = iw_start + sw * dilation;
                    
                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                        shared_input[sh][sw] = input[b * input_stride_batch +
                                               c * input_stride_channel +
                                               ih * input_stride_height +
                                               iw];
                    } else {
                        shared_input[sh][sw] = -std::numeric_limits<scalar_t>::infinity();
                    }
                }
            }
            __syncthreads();
            
            #pragma unroll
            for (int kh = 0; kh < kernel_size; kh++) {
                #pragma unroll
                for (int kw = 0; kw < kernel_size; kw++) {
                    max_val = max(max_val, shared_input[kh][kw]);
                }
            }
        } else {
            #pragma unroll
            for (int kh = 0; kh < kernel_size; kh++) {
                const int ih = h * stride - padding + kh * dilation;
                if (ih >= 0 && ih < input_height) {
                    #pragma unroll
                    for (int kw = 0; kw < kernel_size; kw++) {
                        const int iw = w * stride - padding + kw * dilation;
                        if (iw >= 0 && iw < input_width) {
                            max_val = max(max_val, input[b * input_stride_batch +
                                                      c * input_stride_channel +
                                                      ih * input_stride_height +
                                                      iw]);
                        }
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