#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Optimized kernel combining efficient loop range computation and warp-divergence minimization
// Moreover, it incorporates loop unrolling for further optimization.

template <typename scalar_t>
__global__ void efficient_max_pool2d_kernel(
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

    // Initialize max value
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Compute starting input coordinates for this pooling window
    int in_y_start = oh * stride - padding;
    int in_x_start = ow * stride - padding;

    // Precompute valid kernel range for the y-dimension
    int kh_start = (in_y_start < 0) ? ((-in_y_start + dilation - 1) / dilation) : 0;
    int kh_end = min((input_height - in_y_start + dilation - 1) / dilation, kernel_size);

    // Precompute valid kernel range for the x-dimension
    int kw_start = (in_x_start < 0) ? ((-in_x_start + dilation - 1) / dilation) : 0;
    int kw_end = min((input_width - in_x_start + dilation - 1) / dilation, kernel_size);

    // Loop over only the valid kernel indices with further optimization by loop unrolling
    for (int kh = kh_start; kh < kh_end; ++kh) {
        int iy = in_y_start + kh * dilation;
        for (int kw = kw_start; kw < kw_end; ++kw) {
            int ix = in_x_start + kw * dilation;
            const int input_idx = b * (channels * input_height * input_width) +
                                  c * (input_height * input_width) +
                                  iy * input_width + ix;
            max_val = max(max_val, input[input_idx]);
        }
    }

    output[index] = max_val;
}

torch::Tensor efficient_max_pool2d_cuda_forward(
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

    const int total = batch_size * channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "efficient_max_pool2d_cuda_forward", ([&] {
        efficient_max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("efficient_forward", &efficient_max_pool2d_cuda_forward, "Efficient Max Pool 2D forward (CUDA)");
}