#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// This kernel precomputes the valid kernel boundaries for pooling
// to eliminate divergent branching in the inner loops.

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
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * output_height * output_width;
    if (index >= total) return;

    // Decode the flattened index into (b, c, oh, ow)
    int ow = index % output_width;
    int tmp = index / output_width;
    int oh = tmp % output_height;
    tmp = tmp / output_height;
    int c = tmp % channels;
    int b = tmp / channels;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Compute the top-left corner (input coordinate) for the pooling window
    int in_y = oh * stride - padding;
    int in_x = ow * stride - padding;

    // Precompute valid kernel bounds for the height dimension
    int kh_start = (in_y < 0) ? ((-in_y + dilation - 1) / dilation) : 0;
    int kh_end = (input_height - in_y + dilation - 1) / dilation;
    if (kh_end > kernel_size) kh_end = kernel_size;

    // Precompute valid kernel bounds for the width dimension
    int kw_start = (in_x < 0) ? ((-in_x + dilation - 1) / dilation) : 0;
    int kw_end = (input_width - in_x + dilation - 1) / dilation;
    if (kw_end > kernel_size) kw_end = kernel_size;

    // Loop only over valid kernel indices to avoid runtime conditionals
    for (int kh = kh_start; kh < kh_end; ++kh) {
        int ih = in_y + kh * dilation;
        for (int kw = kw_start; kw < kw_end; ++kw) {
            int iw = in_x + kw * dilation;
            int input_idx = b * (channels * input_height * input_width) +
                            c * (input_height * input_width) +
                            ih * input_width + iw;
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

    const int total = batch_size * channels * output_height * output_width;
    const int threads = 512;  // Experimenting with a larger block size
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA) with optimized block size");
}
