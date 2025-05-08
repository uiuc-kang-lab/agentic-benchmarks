#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void warp_aligned_max_pool2d_kernel(
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
    // Use 32x8 thread block for warp alignment
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int out_x = blockIdx.x * 32 + lane_id;
    const int out_y = (blockIdx.y * 8 + threadIdx.y);
    const int batch_channel = blockIdx.z;
    
    const int batch = batch_channel / channels;
    const int channel = batch_channel % channels;

    // Early exit for entire warp if out of bounds
    if ((out_y >= output_height) || (warp_id >= 8)) {
        return;
    }

    // Calculate input base indices
    const int in_y_base = out_y * stride - padding;
    const int in_x_base = out_x * stride - padding;
    const int batch_offset = batch * channels * input_height * input_width;
    const int channel_offset = channel * input_height * input_width;
    const int input_base = batch_offset + channel_offset;

    scalar_t max_val = -__FLT_MAX__;

    // Handle different kernel sizes with minimal divergence
    if constexpr (KERNEL_SIZE == 2) {
        // All threads in warp process their respective columns
        if (out_x < output_width) {
            #pragma unroll
            for (int ky = 0; ky < 2; ky++) {
                const int in_y = in_y_base + ky * dilation;
                const bool valid_y = (in_y >= 0 && in_y < input_height);
                
                if (valid_y) {
                    const int row_offset = input_base + in_y * input_width;
                    #pragma unroll
                    for (int kx = 0; kx < 2; kx++) {
                        const int in_x = in_x_base + kx * dilation;
                        if (in_x >= 0 && in_x < input_width) {
                            max_val = fmaxf(max_val, __ldg(&input[row_offset + in_x]));
                        }
                    }
                }
            }
        }
    }
    else if constexpr (KERNEL_SIZE == 3) {
        if (out_x < output_width) {
            #pragma unroll
            for (int ky = 0; ky < 3; ky++) {
                const int in_y = in_y_base + ky * dilation;
                const bool valid_y = (in_y >= 0 && in_y < input_height);
                
                if (valid_y) {
                    const int row_offset = input_base + in_y * input_width;
                    #pragma unroll
                    for (int kx = 0; kx < 3; kx++) {
                        const int in_x = in_x_base + kx * dilation;
                        if (in_x >= 0 && in_x < input_width) {
                            max_val = fmaxf(max_val, __ldg(&input[row_offset + in_x]));
                        }
                    }
                }
            }
        }
    }
    else {
        if (out_x < output_width) {
            for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                const int in_y = in_y_base + ky * dilation;
                const bool valid_y = (in_y >= 0 && in_y < input_height);
                
                if (valid_y) {
                    const int row_offset = input_base + in_y * input_width;
                    for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                        const int in_x = in_x_base + kx * dilation;
                        if (in_x >= 0 && in_x < input_width) {
                            max_val = fmaxf(max_val, __ldg(&input[row_offset + in_x]));
                        }
                    }
                }
            }
        }
    }

    // Write output only if within bounds
    if (out_x < output_width) {
        const int out_idx = batch_channel * output_height * output_width + 
                           out_y * output_width + out_x;
        output[out_idx] = max_val;
    }
}

torch::Tensor max_pool2d_cuda_forward(
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

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    dim3 threads(32, 8);
    dim3 blocks(
        (output_width + 31) / 32,
        (output_height + 7) / 8,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            warp_aligned_max_pool2d_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        }
        else if (kernel_size == 3) {
            warp_aligned_max_pool2d_kernel<scalar_t, 3><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        }
        else {
            warp_aligned_max_pool2d_kernel<scalar_t, -1><<<blocks, threads>>>(
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