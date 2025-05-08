#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// This kernel combines best practices from two implementations. It uses a 2D thread block for the spatial dimensions and maps the batch and channel dimensions to the third grid dimension. 
// It provides compile-time unrolling for the common cases (kernel_size==2 or 3) and falls back to a runtime parameter for other kernel sizes.

template <typename scalar_t, int KERNEL_SIZE>
__global__ void efficient_max_pool2d_kernel(
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
    const int dilation,
    const int runtime_kernel_size  // used only when KERNEL_SIZE <= 0
) {
    // Map thread to output spatial coordinates
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    // The third grid dimension is the combined (batch, channel) index
    const int bc = blockIdx.z;
    if (ow >= output_width || oh >= output_height) return;

    const int b = bc / channels;
    const int c = bc % channels;

    // Compute base pointer for the (b, c) slice
    const int input_batch_stride = channels * input_height * input_width;
    const int input_channel_stride = input_height * input_width;
    const int base_idx = b * input_batch_stride + c * input_channel_stride;

    // Initialize max value (assuming floating point: works for half/float/double)
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Compute starting indices in the input
    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;

    // Determine the effective kernel size. For specialized cases (2 or 3) the template parameter is used;
    // otherwise, use the runtime value.
    const int effectiveKernelSize = (KERNEL_SIZE > 0 ? KERNEL_SIZE : runtime_kernel_size);

    if constexpr (KERNEL_SIZE == 2) {
        #pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
            const int ih = ih_start + kh * dilation;
            if (ih < 0 || ih >= input_height) continue;
            const int row_idx = base_idx + ih * input_width;
            #pragma unroll
            for (int kw = 0; kw < 2; ++kw) {
                const int iw = iw_start + kw * dilation;
                if (iw < 0 || iw >= input_width) continue;
                max_val = max(max_val, __ldg(&input[row_idx + iw]));
            }
        }
    } else if constexpr (KERNEL_SIZE == 3) {
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            const int ih = ih_start + kh * dilation;
            if (ih < 0 || ih >= input_height) continue;
            const int row_idx = base_idx + ih * input_width;
            #pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                const int iw = iw_start + kw * dilation;
                if (iw < 0 || iw >= input_width) continue;
                max_val = max(max_val, __ldg(&input[row_idx + iw]));
            }
        }
    } else {
        // General case for arbitrary kernel sizes
        for (int kh = 0; kh < effectiveKernelSize; ++kh) {
            const int ih = ih_start + kh * dilation;
            if (ih < 0 || ih >= input_height) continue;
            const int row_idx = base_idx + ih * input_width;
            for (int kw = 0; kw < effectiveKernelSize; ++kw) {
                const int iw = iw_start + kw * dilation;
                if (iw < 0 || iw >= input_width) continue;
                max_val = max(max_val, __ldg(&input[row_idx + iw]));
            }
        }
    }

    const int out_idx = b * (channels * output_height * output_width) + c * (output_height * output_width) + oh * output_width + ow;
    output[out_idx] = max_val;
}

// Host launcher function

torch::Tensor efficient_max_pool2d_cuda_forward(
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
    
    // Compute output dimensions
    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Use a 2D thread block to cover the output spatial dimensions and map batch*channels to grid.z
    dim3 threads(16, 16);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "efficient_max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            efficient_max_pool2d_kernel<scalar_t, 2><<<blocks, threads>>>(
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
                dilation,
                0  // dummy parameter, not used
            );
        } else if (kernel_size == 3) {
            efficient_max_pool2d_kernel<scalar_t, 3><<<blocks, threads>>>(
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
                dilation,
                0  // dummy parameter
            );
        } else {
            // For arbitrary kernel sizes, pass kernel_size as a runtime parameter
            efficient_max_pool2d_kernel<scalar_t, -1><<<blocks, threads>>>(
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
                dilation,
                kernel_size
            );
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &efficient_max_pool2d_cuda_forward, "Efficient Max Pool 2D forward (CUDA)");
}
