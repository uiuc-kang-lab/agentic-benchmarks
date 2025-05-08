#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_tiled_kernel(
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
    // 32x8 thread block for better occupancy
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = blockIdx.z;

    if (ow >= output_width || oh >= output_height) return;

    const int b = bc / channels;
    const int c = bc % channels;
    const int base_idx = b * channels * input_height * input_width
                       + c * input_height * input_width;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const int ih_base = oh * stride - padding;
    const int iw_base = ow * stride - padding;

    // Loop with implicit boundary checks via unsigned comparison
    if constexpr (KERNEL_SIZE == 2) {
        #pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
            const unsigned ih = ih_base + kh * dilation;
            if (ih < input_height) {
                #pragma unroll
                for (int kw = 0; kw < 2; ++kw) {
                    const unsigned iw = iw_base + kw * dilation;
                    if (iw < input_width) {
                        max_val = fmaxf(max_val, __ldg(&input[base_idx + ih * input_width + iw]));
                    }
                }
            }
        }
    } else if constexpr (KERNEL_SIZE == 3) {
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            const unsigned ih = ih_base + kh * dilation;
            if (ih < input_height) {
                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    const unsigned iw = iw_base + kw * dilation;
                    if (iw < input_width) {
                        max_val = fmaxf(max_val, __ldg(&input[base_idx + ih * input_width + iw]));
                    }
                }
            }
        }
    } else {
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            const unsigned ih = ih_base + kh * dilation;
            if (ih < input_height) {
                for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                    const unsigned iw = iw_base + kw * dilation;
                    if (iw < input_width) {
                        max_val = fmaxf(max_val, __ldg(&input[base_idx + ih * input_width + iw]));
                    }
                }
            }
        }
    }

    output[b * channels * output_height * output_width
         + c * output_height * output_width
         + oh * output_width
         + ow] = max_val;
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

    dim3 threads(32, 8);  // 256 threads per block
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            max_pool2d_tiled_kernel<scalar_t, 2><<<blocks, threads>>>(
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
            max_pool2d_tiled_kernel<scalar_t, 3><<<blocks, threads>>>(
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
            max_pool2d_tiled_kernel<scalar_t, -1><<<blocks, threads>>>(
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
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with optimized tiling (CUDA)");
}
