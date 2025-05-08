#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Generic kernel for arbitrary kernel sizes
template <typename scalar_t>
__global__ void max_pool2d_generic_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    int ow = output_idx % output_width;
    int oh = (output_idx / output_width) % output_height;
    int c  = (output_idx / (output_width * output_height)) % channels;
    int b  = output_idx / (output_width * output_height * channels);

    int input_batch_offset = b * (channels * input_height * input_width);
    int input_channel_offset = c * (input_height * input_width);
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Iterate over the pooling window
    for (int kh = 0; kh < kernel_size; kh++) {
        int ih = oh * stride - padding + kh * dilation;
        if (ih < 0 || ih >= input_height) continue;
        int ih_offset = ih * input_width;
        for (int kw = 0; kw < kernel_size; kw++) {
            int iw = ow * stride - padding + kw * dilation;
            if (iw < 0 || iw >= input_width) continue;
            int idx = input_batch_offset + input_channel_offset + ih_offset + iw;
            max_val = max(max_val, __ldg(&input[idx]));
        }
    }
    output[output_idx] = max_val;
}

// Specialized kernel for fixed kernel sizes (2 and 3) with full loop unrolling
template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_blocksize_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int stride,
    int padding,
    int dilation
) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    int ow = output_idx % output_width;
    int oh = (output_idx / output_width) % output_height;
    int c  = (output_idx / (output_width * output_height)) % channels;
    int b  = output_idx / (output_width * output_height * channels);

    int input_batch_offset = b * (channels * input_height * input_width);
    int input_channel_offset = c * (input_height * input_width);
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    int ih_base = oh * stride - padding;
    int iw_base = ow * stride - padding;

    if constexpr (KERNEL_SIZE == 2) {
        // Fully unroll 2x2 pooling window
        if (ih_base >= 0 && ih_base < input_height && iw_base >= 0 && iw_base < input_width) {
            int idx = input_batch_offset + input_channel_offset + ih_base * input_width + iw_base;
            max_val = __ldg(&input[idx]);
        }
        if (ih_base >= 0 && ih_base < input_height && (iw_base + dilation) >= 0 && (iw_base + dilation) < input_width) {
            int idx = input_batch_offset + input_channel_offset + ih_base * input_width + (iw_base + dilation);
            max_val = max(max_val, __ldg(&input[idx]));
        }
        if ((ih_base + dilation) >= 0 && (ih_base + dilation) < input_height && iw_base >= 0 && iw_base < input_width) {
            int idx = input_batch_offset + input_channel_offset + (ih_base + dilation) * input_width + iw_base;
            max_val = max(max_val, __ldg(&input[idx]));
        }
        if ((ih_base + dilation) >= 0 && (ih_base + dilation) < input_height && (iw_base + dilation) >= 0 && (iw_base + dilation) < input_width) {
            int idx = input_batch_offset + input_channel_offset + (ih_base + dilation) * input_width + (iw_base + dilation);
            max_val = max(max_val, __ldg(&input[idx]));
        }
    } else if constexpr (KERNEL_SIZE == 3) {
        // Fully unroll 3x3 pooling window
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            int ih = ih_base + i * dilation;
            if (ih < 0 || ih >= input_height) continue;
            int ih_offset = ih * input_width;
            #pragma unroll
            for (int j = 0; j < 3; j++) {
                int iw = iw_base + j * dilation;
                if (iw < 0 || iw >= input_width) continue;
                int idx = input_batch_offset + input_channel_offset + ih_offset + iw;
                max_val = max(max_val, __ldg(&input[idx]));
            }
        }
    }

    output[output_idx] = max_val;
}

// Host function with tunable block size
torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int block_size  // allows experimentation with block sizes (e.g., 32, 64, 128, 256, 512)
) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    // Calculate output dimensions
    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    int total_elements = batch_size * channels * output_height * output_width;
    int blocks = (total_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            max_pool2d_blocksize_kernel<scalar_t, 2><<<blocks, block_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                channels,
                input_height,
                input_width,
                output_height,
                output_width,
                stride,
                padding,
                dilation
            );
        } else if (kernel_size == 3) {
            max_pool2d_blocksize_kernel<scalar_t, 3><<<blocks, block_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                channels,
                input_height,
                input_width,
                output_height,
                output_width,
                stride,
                padding,
                dilation
            );
        } else {
            // Fallback kernel for arbitrary kernel sizes
            max_pool2d_generic_kernel<scalar_t><<<blocks, block_size>>>(
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
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with tunable block sizes (CUDA)");
}
