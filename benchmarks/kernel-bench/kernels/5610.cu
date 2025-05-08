#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_coalesced_kernel(
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
    // Use shared memory to store input data for the block
    __shared__ scalar_t shared_input[32][34];  // 32 width + 2 padding

    const int tid = threadIdx.x;
    const int block_x = blockIdx.x * 32;
    const int oh = blockIdx.y;
    const int bc = blockIdx.z;
    
    const int b = bc / channels;
    const int c = bc % channels;

    // Calculate input base indices
    const int ih_start = oh * stride - padding;
    const int input_b_offset = b * channels * input_height * input_width;
    const int input_c_offset = c * input_height * input_width;

    if constexpr (KERNEL_SIZE == 2) {
        #pragma unroll
        for (int kh = 0; kh < 2; kh++) {
            const int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                const int input_row = input_b_offset + input_c_offset + ih * input_width;
                
                // Coalesced load into shared memory
                for (int i = tid; i < 34 && (block_x + i - 1) < input_width; i += 32) {
                    const int iw = block_x + i - 1;
                    if (iw >= 0 && iw < input_width) {
                        shared_input[tid][i] = __ldg(&input[input_row + iw]);
                    } else {
                        shared_input[tid][i] = -__FLT_MAX__;
                    }
                }
                __syncthreads();

                // Process output elements
                if (tid < output_width - block_x) {
                    scalar_t max_val = -__FLT_MAX__;
                    #pragma unroll
                    for (int kw = 0; kw < 2; kw++) {
                        max_val = fmaxf(max_val, shared_input[tid][tid + kw]);
                    }
                    const int out_idx = bc * output_height * output_width + oh * output_width + block_x + tid;
                    if (kh == 0) {
                        output[out_idx] = max_val;
                    } else {
                        output[out_idx] = fmaxf(output[out_idx], max_val);
                    }
                }
                __syncthreads();
            }
        }
    } else if constexpr (KERNEL_SIZE == 3) {
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            const int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                const int input_row = input_b_offset + input_c_offset + ih * input_width;
                
                // Coalesced load into shared memory
                for (int i = tid; i < 34 && (block_x + i - 1) < input_width; i += 32) {
                    const int iw = block_x + i - 1;
                    if (iw >= 0 && iw < input_width) {
                        shared_input[tid][i] = __ldg(&input[input_row + iw]);
                    } else {
                        shared_input[tid][i] = -__FLT_MAX__;
                    }
                }
                __syncthreads();

                // Process output elements
                if (tid < output_width - block_x) {
                    scalar_t max_val = -__FLT_MAX__;
                    #pragma unroll
                    for (int kw = 0; kw < 3; kw++) {
                        max_val = fmaxf(max_val, shared_input[tid][tid + kw]);
                    }
                    const int out_idx = bc * output_height * output_width + oh * output_width + block_x + tid;
                    if (kh == 0) {
                        output[out_idx] = max_val;
                    } else {
                        output[out_idx] = fmaxf(output[out_idx], max_val);
                    }
                }
                __syncthreads();
            }
        }
    } else {
        for (int kh = 0; kh < KERNEL_SIZE; kh++) {
            const int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                const int input_row = input_b_offset + input_c_offset + ih * input_width;
                
                // Coalesced load into shared memory
                for (int i = tid; i < 34 && (block_x + i - 1) < input_width; i += 32) {
                    const int iw = block_x + i - 1;
                    if (iw >= 0 && iw < input_width) {
                        shared_input[tid][i] = __ldg(&input[input_row + iw]);
                    } else {
                        shared_input[tid][i] = -__FLT_MAX__;
                    }
                }
                __syncthreads();

                // Process output elements
                if (tid < output_width - block_x) {
                    scalar_t max_val = -__FLT_MAX__;
                    for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                        max_val = fmaxf(max_val, shared_input[tid][tid + kw]);
                    }
                    const int out_idx = bc * output_height * output_width + oh * output_width + block_x + tid;
                    if (kh == 0) {
                        output[out_idx] = max_val;
                    } else {
                        output[out_idx] = fmaxf(output[out_idx], max_val);
                    }
                }
                __syncthreads();
            }
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

    const dim3 threads(32);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        output_height,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            max_pool2d_coalesced_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else if (kernel_size == 3) {
            max_pool2d_coalesced_kernel<scalar_t, 3><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else {
            max_pool2d_coalesced_kernel<scalar_t, -1><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with coalesced memory access (CUDA)");
}