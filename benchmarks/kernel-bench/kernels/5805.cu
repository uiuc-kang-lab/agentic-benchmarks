#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Using __ldg() for read-only data and aligning memory accesses to 128-bit boundaries

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
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    int in_y_start = oh * stride - padding;
    int in_x_start = ow * stride - padding;

    int kh_start = (in_y_start < 0) ? ((-in_y_start + dilation - 1) / dilation) : 0;
    int kh_end = (input_height - in_y_start + dilation - 1) / dilation;
    if (kh_end > kernel_size) kh_end = kernel_size;

    int kw_start = (in_x_start < 0) ? ((-in_x_start + dilation - 1) / dilation) : 0;
    int kw_end = (input_width - in_x_start + dilation - 1) / dilation;
    if (kw_end > kernel_size) kw_end = kernel_size;

    for (int kh = kh_start; kh < kh_end; ++kh) {
        int iy = in_y_start + kh * dilation;
        
        for (int kw = kw_start; kw < kw_end; ++kw) {
            int ix = in_x_start + kw * dilation;

            const int input_idx = b * (channels * input_height * input_width) +
                                c * (input_height * input_width) +
                                iy * input_width + ix;

            scalar_t val = __ldg(&input[input_idx]);
            max_val = max(max_val, val);
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
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int total = batch_size * channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA) with ldg optimization");
}
