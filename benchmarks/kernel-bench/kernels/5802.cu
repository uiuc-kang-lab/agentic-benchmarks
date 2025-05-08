#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void max_pool2d_kernel(
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
    extern __shared__ scalar_t shared_input[];
    
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * channels * output_height * output_width;
    if (index >= total) return;

    const int ow = index % output_width;
    int temp = index / output_width;
    const int oh = temp % output_height;
    temp = temp / output_height;
    const int c = temp % channels;
    const int b = temp / channels;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    const int in_y_start = oh * stride - padding;
    const int in_x_start = ow * stride - padding;

    const int kh_start = max(0, (-in_y_start + dilation - 1) / dilation);
    const int kh_end = min(kernel_size, (input_height - in_y_start + dilation - 1) / dilation);
    const int kw_start = max(0, (-in_x_start + dilation - 1) / dilation);
    const int kw_end = min(kernel_size, (input_width - in_x_start + dilation - 1) / dilation);

    const int input_base = b * (channels * input_height * input_width) +
                          c * (input_height * input_width);

    #pragma unroll
    for (int kh = kh_start; kh < kh_end; ++kh) {
        const int iy = in_y_start + kh * dilation;
        #pragma unroll
        for (int kw = kw_start; kw < kw_end; ++kw) {
            const int ix = in_x_start + kw * dilation;
            const int input_idx = input_base + iy * input_width + ix;
            max_val = max(max_val, input[input_idx]);
        }
    }

    output[index] = max_val;
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

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;
    
    const int shared_mem_size = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    }));

    return output;
}