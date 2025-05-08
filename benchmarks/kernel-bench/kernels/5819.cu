#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>


template <typename scalar_t>
__global__ void max_pool2d_kernel_combined(
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
    // Compute the global output index
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * channels * output_height * output_width;
    if (index >= total) return;

    // Decode index into (b, c, oh, ow)
    int ow = index % output_width;
    int temp = index / output_width;
    int oh = temp % output_height;
    temp = temp / output_height;
    int c = temp % channels;
    int b = temp / channels;

    // Precompute the base index for the current (b, c) slice
    const int in_image_size = input_height * input_width;
    const int base_idx = b * channels * in_image_size + c * in_image_size;

    // Compute the top-left coordinate of the pooling window in the input
    int in_y_start = oh * stride - padding;
    int in_x_start = ow * stride - padding;

    // Precompute the valid kernel boundaries to avoid conditionals inside the loops
    int kh_start = (in_y_start < 0) ? ((-in_y_start + dilation - 1) / dilation) : 0;
    int kh_end = (input_height - in_y_start + dilation - 1) / dilation;
    if (kh_end > kernel_size) kh_end = kernel_size;

    int kw_start = (in_x_start < 0) ? ((-in_x_start + dilation - 1) / dilation) : 0;
    int kw_end = (input_width - in_x_start + dilation - 1) / dilation;
    if (kw_end > kernel_size) kw_end = kernel_size;

    // Initialize max value
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Iterate over the valid region of the pooling window
    for (int kh = kh_start; kh < kh_end; ++kh) {
        int iy = in_y_start + kh * dilation;
        int row_offset = iy * input_width;  // Pre-calc the row offset
        for (int kw = kw_start; kw < kw_end; ++kw) {
            int ix = in_x_start + kw * dilation;
            int input_idx = base_idx + row_offset + ix;
            max_val = max(max_val, input[input_idx]);
        }
    }

    // Write the result for this output element
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

    // Calculate output dimensions
    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int total = batch_size * channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel_combined<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA) - combined optimizations");
}
