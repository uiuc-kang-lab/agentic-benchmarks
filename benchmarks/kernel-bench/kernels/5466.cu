#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__device__ inline int calculate_input_index(const int b, const int c, const int h, const int w,
                                          const int channels, const int height, const int width) {
    return b * (channels * height * width) +
           c * (height * width) +
           h * width +
           w;
}

template <typename scalar_t, int KERNEL_SIZE>
__global__ void hybrid_maxpool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* output,
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

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    __shared__ scalar_t shared_input[32][32];

    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;

    #pragma unroll
    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
        const int ih = ih_start + kh * dilation;
        
        #pragma unroll
        for (int kw = 0; kw < KERNEL_SIZE; kw++) {
            const int iw = iw_start + kw * dilation;
            
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                const int input_idx = calculate_input_index(b, c, ih, iw, channels, input_height, input_width);
                max_val = max(max_val, __ldg(&input[input_idx]));
            }
        }
    }

    output[output_idx] = max_val;
}

torch::Tensor hybrid_maxpool2d_cuda_forward(
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

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hybrid_maxpool2d_cuda_forward", ([&] {
        switch(kernel_size) {
            case 2:
                hybrid_maxpool2d_kernel<scalar_t, 2><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation);
                break;
            case 3:
                hybrid_maxpool2d_kernel<scalar_t, 3><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation);
                break;
            default:
                hybrid_maxpool2d_kernel<scalar_t, -1><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation);
        }
    }));

    return output;
}