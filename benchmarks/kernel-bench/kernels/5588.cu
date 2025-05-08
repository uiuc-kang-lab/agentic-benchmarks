#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_unrolled_kernel(
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
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    const int input_batch_stride = channels * input_height * input_width;
    const int input_channel_stride = input_height * input_width;
    const int base_idx = b * input_batch_stride + c * input_channel_stride;
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;

    if constexpr (KERNEL_SIZE == 2) {
        #pragma unroll
        for (int kh = 0; kh < 2; kh++) {
            const int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                const int row_idx = base_idx + ih * input_width;
                #pragma unroll
                for (int kw = 0; kw < 2; kw++) {
                    const int iw = iw_start + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        max_val = max(max_val, __ldg(&input[row_idx + iw]));
                    }
                }
            }
        }
    } else if constexpr (KERNEL_SIZE == 3) {
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            const int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                const int row_idx = base_idx + ih * input_width;
                #pragma unroll
                for (int kw = 0; kw < 3; kw++) {
                    const int iw = iw_start + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        max_val = max(max_val, __ldg(&input[row_idx + iw]));
                    }
                }
            }
        }
    } else {
        for (int kh = 0; kh < KERNEL_SIZE; kh++) {
            const int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                const int row_idx = base_idx + ih * input_width;
                #pragma unroll 4
                for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                    const int iw = iw_start + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        max_val = max(max_val, __ldg(&input[row_idx + iw]));
                    }
                }
            }
        }
    }

    output[output_idx] = max_val;
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

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            max_pool2d_unrolled_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else if (kernel_size == 3) {
            max_pool2d_unrolled_kernel<scalar_t, 3><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else {
            max_pool2d_unrolled_kernel<scalar_t, -1><<<blocks, threads>>>(
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