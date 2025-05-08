#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_warp_kernel(
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
    // Use 32x8 thread block configuration for warp alignment
    const int tx = threadIdx.x;    // 0-31 (warp aligned)
    const int ty = threadIdx.y;    // 0-7
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // Compute output coordinates
    const int ow = bx * 32 + tx;
    const int oh = by * 8 + ty;
    
    // Early exit for entire warp if outside output bounds
    if (by * 8 >= output_height) return;
    if (bx * 32 >= output_width) return;

    const int b = bz / channels;
    const int c = bz % channels;

    // Pre-compute input base indices and bounds (uniform across warp)
    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;
    
    // Input base offset
    const int input_idx_base = b * channels * input_height * input_width + 
                              c * input_height * input_width;

    scalar_t max_val = -__FLT_MAX__;

    // Compute valid ranges for entire warp
    const int ih_end = min(ih_start + (KERNEL_SIZE - 1) * dilation + 1, input_height);
    const int iw_end = min(iw_start + (KERNEL_SIZE - 1) * dilation + 1, input_width);
    const int ih_valid_start = max(0, ih_start);
    const int iw_valid_start = max(0, iw_start);

    if (ow < output_width && oh < output_height) {
        if constexpr (KERNEL_SIZE == 2) {
            // Special handling for 2x2 kernel - attempt vectorized loads
            if (dilation == 1 && stride == 2 && iw_valid_start + 1 < input_width) {
                const int row1_idx = input_idx_base + ih_valid_start * input_width + iw_valid_start;
                const int row2_idx = row1_idx + input_width;
                
                if (ih_valid_start + 1 < input_height) {
                    const float2 row1 = *reinterpret_cast<const float2*>(&input[row1_idx]);
                    const float2 row2 = *reinterpret_cast<const float2*>(&input[row2_idx]);
                    max_val = max(max(row1.x, row1.y), max(row2.x, row2.y));
                } else {
                    const float2 row1 = *reinterpret_cast<const float2*>(&input[row1_idx]);
                    max_val = max(row1.x, row1.y);
                }
            } else {
                #pragma unroll
                for (int kh = 0; kh < 2; kh++) {
                    const int ih = ih_valid_start + kh * dilation;
                    if (ih < input_height) {
                        const int row_idx = input_idx_base + ih * input_width;
                        #pragma unroll
                        for (int kw = 0; kw < 2; kw++) {
                            const int iw = iw_valid_start + kw * dilation;
                            if (iw < input_width) {
                                max_val = max(max_val, __ldg(&input[row_idx + iw]));
                            }
                        }
                    }
                }
            }
        } else if constexpr (KERNEL_SIZE == 3) {
            // Uniform bounds checking for 3x3
            const bool row_valid = (ih_valid_start < ih_end);
            const bool col_valid = (iw_valid_start < iw_end);
            
            if (row_valid && col_valid) {
                #pragma unroll
                for (int kh = 0; kh < 3; kh++) {
                    const int ih = ih_valid_start + kh * dilation;
                    if (ih < input_height) {
                        const int row_idx = input_idx_base + ih * input_width;
                        #pragma unroll
                        for (int kw = 0; kw < 3; kw++) {
                            const int iw = iw_valid_start + kw * dilation;
                            if (iw < input_width) {
                                max_val = max(max_val, __ldg(&input[row_idx + iw]));
                            }
                        }
                    }
                }
            }
        }

        // Write output
        const int out_idx = b * channels * output_height * output_width +
                           c * output_height * output_width +
                           oh * output_width + ow;
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

    const dim3 threads(32, 8);
    const dim3 blocks(
        (output_width + 31) / 32,
        (output_height + 7) / 8,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            max_pool2d_warp_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else if (kernel_size == 3) {
            max_pool2d_warp_kernel<scalar_t, 3><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else {
            max_pool2d_warp_kernel<scalar_t, -1><<<blocks, threads>>>(
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