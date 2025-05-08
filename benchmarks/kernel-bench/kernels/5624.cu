#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Helper function to align pointer to 128 bits
inline int align_up(int value, int alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

// Kernel template to handle different kernel sizes
template <typename scalar_t, int KERNEL_SIZE>
__global__ void aligned_max_pool2d_kernel(
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
    const int dynamic_ksize
) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = blockIdx.z;

    if (ow >= output_width || oh >= output_height) return;

    const int b = bc / channels;
    const int c = bc % channels;

    const int aligned_input_width = align_up(input_width, 4);

    const int input_offset = b * channels * input_height * aligned_input_width 
                           + c * input_height * aligned_input_width;
    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const int kernel_size = KERNEL_SIZE > 0 ? KERNEL_SIZE : dynamic_ksize;

    if constexpr (KERNEL_SIZE == 2) {
        #pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
            const int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                #pragma unroll
                for (int kw = 0; kw < 2; ++kw) {
                    const int iw = iw_start + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        max_val = max(max_val, __ldg(&input[input_offset + ih * aligned_input_width + iw]));
                    }
                }
            }
        }
    } else if constexpr (KERNEL_SIZE == 3) {
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            const int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    const int iw = iw_start + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        max_val = max(max_val, __ldg(&input[input_offset + ih * aligned_input_width + iw]));
                    }
                }
            }
        }
    } else {
        for (int kh = 0; kh < kernel_size; ++kh) {
            const int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int iw = iw_start + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        max_val = max(max_val, __ldg(&input[input_offset + ih * aligned_input_width + iw]));
                    }
                }
            }
        }
    }

    output[bc * output_height * output_width + oh * output_width + ow] = max_val;
}

// Host function to launch the kernel
torch::Tensor max_pool2d_aligned_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_h = input.size(2);
    const auto input_w = input.size(3);

    const auto output_h = ((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_w = ((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_h, output_w}, input.options());

    dim3 threads(16, 16);
    dim3 blocks(
        (output_w + threads.x - 1) / threads.x,
        (output_h + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "aligned_pool_forward", [&] {
        if (kernel_size == 2) {
            aligned_max_pool2d_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_h, input_w,
                output_h, output_w,
                stride, padding, dilation, 0);
        } else if (kernel_size == 3) {
            aligned_max_pool2d_kernel<scalar_t, 3><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_h, input_w,
                output_h, output_w,
                stride, padding, dilation, 0);
        } else {
            aligned_max_pool2d_kernel<scalar_t, -1><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_h, input_w,
                output_h, output_w,
                stride, padding, dilation, kernel_size);
        }
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_aligned_forward, "Aligned Max Pool 2D forward (CUDA)");
}
