#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_optimized_kernel(
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
    // Spatial dimensions mapped to 2D block for coalescing
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;
    if (ow >= output_width || oh >= output_height) return;

    // Decompose batch-channel index
    int b = bc / channels;
    int c = bc % channels;
    
    // Precompute input offsets
    const int input_base = b * channels * input_height * input_width
                        + c * input_height * input_width;
    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;

    // Optimized loop with boundary checks
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    if constexpr (KERNEL_SIZE == 2) {
        #pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
            int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                int row_off = ih * input_width;
                #pragma unroll
                for (int kw = 0; kw < 2; ++kw) {
                    int iw = iw_start + kw * dilation;
                    if (iw >= 0 && iw < input_width)
                        max_val = fmaxf(max_val, __ldg(input + input_base + row_off + iw));
                }
            }
        }
    }
    else if constexpr (KERNEL_SIZE == 3) {
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                int row_off = ih * input_width;
                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    int iw = iw_start + kw * dilation;
                    if (iw >= 0 && iw < input_width)
                        max_val = fmaxf(max_val, __ldg(input + input_base + row_off + iw));
                }
            }
        }
    }
    else {
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                int row_off = ih * input_width;
                for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                    int iw = iw_start + kw * dilation;
                    if (iw >= 0 && iw < input_width)
                        max_val = fmaxf(max_val, __ldg(input + input_base + row_off + iw));
                }
            }
        }
    }

    // Coalesced write with spatial locality
    output[b * channels * output_height * output_width
         + c * output_height * output_width
         + oh * output_width
         + ow] = max_val;
}

torch::Tensor max_pool2d_optimized_forward(
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
    const auto output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const auto output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    dim3 threads(16, 16);  // Better memory coalescing
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels  // Natural bc grouping
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool_forward", ([&] {
        if (kernel_size == 2) {
            max_pool2d_optimized_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation);
        } else if (kernel_size == 3) {
            max_pool2d_optimized_kernel<scalar_t, 3><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation);
        } else {
            max_pool2d_optimized_kernel<scalar_t, -1><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation);
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_optimized_forward, "Max Pool 2D Optimized forward (CUDA)");
}