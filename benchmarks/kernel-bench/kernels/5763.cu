#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t, int STATIC_KERNEL_SIZE=-1>
__global__ void optimized_max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    extern __shared__ char shared_memory[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_memory);

    const int tid = threadIdx.x;
    const int output_idx = blockIdx.x * blockDim.x + tid;
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    if constexpr (STATIC_KERNEL_SIZE > 0) {
        #pragma unroll
        for (int kh = 0; kh < STATIC_KERNEL_SIZE; kh++) {
            #pragma unroll
            for (int kw = 0; kw < STATIC_KERNEL_SIZE; kw++) {
                const int ih = oh * stride - padding + kh * dilation;
                const int iw = ow * stride - padding + kw * dilation;

                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    const int input_idx = b * (channels * input_height * input_width) +
                                        c * (input_height * input_width) +
                                        ih * input_width +
                                        iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    } else {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                const int ih = oh * stride - padding + kh * dilation;
                const int iw = ow * stride - padding + kw * dilation;

                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    const int input_idx = b * (channels * input_height * input_width) +
                                        c * (input_height * input_width) +
                                        ih * input_width +
                                        iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    }

    output[output_idx] = max_val;
}

torch::Tensor max_pool2d_cuda_forward_optimized(
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

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward_optimized", ([&] {
        switch(kernel_size) {
            case 2:
                optimized_max_pool2d_kernel<scalar_t, 2><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, kernel_size,
                    stride, padding, dilation
                );
                break;
            case 3:
                optimized_max_pool2d_kernel<scalar_t, 3><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, kernel_size,
                    stride, padding, dilation
                );
                break;
            default:
                optimized_max_pool2d_kernel<scalar_t, -1><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, kernel_size,
                    stride, padding, dilation
                );
        }
    }));

    return output;
}