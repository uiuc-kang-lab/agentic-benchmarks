#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// This kernel uses a grid-stride loop over all output elements and supports loop unrolling for common kernel sizes (2 and 3).
// The host function accepts a block_size parameter so that users can experiment with different block sizes (e.g. 32, 64, 128, 256, 512) to determine the optimal configuration.

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_kernel_tuned(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size, // runtime kernel size if KERNEL_SIZE <= 0
    const int stride,
    const int padding,
    const int dilation,
    const int total_elements
) {
    // Grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
        // Decompose linear index into (b, c, oh, ow)
        int ow = idx % output_width;
        int oh = (idx / output_width) % output_height;
        int c  = (idx / (output_width * output_height)) % channels;
        int b  = idx / (output_width * output_height * channels);

        const int input_batch_stride = channels * input_height * input_width;
        const int input_channel_stride = input_height * input_width;
        const int base_idx = b * input_batch_stride + c * input_channel_stride;

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        const int ih_start = oh * stride - padding;
        const int iw_start = ow * stride - padding;

        // Determine effective kernel size
        int effective_kernel = (KERNEL_SIZE > 0 ? KERNEL_SIZE : kernel_size);

        if constexpr (KERNEL_SIZE == 2) {
            #pragma unroll
            for (int kh = 0; kh < 2; ++kh) {
                int ih = ih_start + kh * dilation;
                if (ih < 0 || ih >= input_height) continue;
                int row_idx = base_idx + ih * input_width;
                #pragma unroll
                for (int kw = 0; kw < 2; ++kw) {
                    int iw = iw_start + kw * dilation;
                    if (iw < 0 || iw >= input_width) continue;
                    max_val = max(max_val, __ldg(&input[row_idx + iw]));
                }
            }
        } else if constexpr (KERNEL_SIZE == 3) {
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                int ih = ih_start + kh * dilation;
                if (ih < 0 || ih >= input_height) continue;
                int row_idx = base_idx + ih * input_width;
                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    int iw = iw_start + kw * dilation;
                    if (iw < 0 || iw >= input_width) continue;
                    max_val = max(max_val, __ldg(&input[row_idx + iw]));
                }
            }
        } else {
            // Generic case for arbitrary kernel sizes
            for (int kh = 0; kh < effective_kernel; ++kh) {
                int ih = ih_start + kh * dilation;
                if (ih < 0 || ih >= input_height) continue;
                int row_idx = base_idx + ih * input_width;
                for (int kw = 0; kw < effective_kernel; ++kw) {
                    int iw = iw_start + kw * dilation;
                    if (iw < 0 || iw >= input_width) continue;
                    max_val = max(max_val, __ldg(&input[row_idx + iw]));
                }
            }
        }

        output[idx] = max_val;
    }
}

// Host launcher with a tunable block_size parameter
// Users can experiment with different block sizes (e.g., 32, 64, 128, 256, 512) to optimize performance on their hardware.

torch::Tensor tuned_blocksize_max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int block_size  // block size for kernel launch
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());
    int total_elements = batch_size * channels * output_height * output_width;

    // Compute grid dimension based on the chosen block size
    int grid_size = (total_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tuned_blocksize_max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            max_pool2d_kernel_tuned<scalar_t, 2><<<grid_size, block_size>>>(
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
                dilation,
                total_elements
            );
        } else if (kernel_size == 3) {
            max_pool2d_kernel_tuned<scalar_t, 3><<<grid_size, block_size>>>(
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
                dilation,
                total_elements
            );
        } else {
            max_pool2d_kernel_tuned<scalar_t, -1><<<grid_size, block_size>>>(
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
                dilation,
                total_elements
            );
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &tuned_blocksize_max_pool2d_cuda_forward, "Tuned Block Size Max Pool 2D forward (CUDA)");
}
