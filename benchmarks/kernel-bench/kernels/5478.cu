#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

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
    const int total_outputs = batch_size * channels * output_height * output_width;
    if (output_idx >= total_outputs) return;

    // Calculate position using warp-aligned indexing
    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    // Pre-compute valid ranges for pooling window
    const int ih_start = max(oh * stride - padding, 0);
    const int ih_end = min(oh * stride - padding + kernel_size * dilation, input_height);
    const int iw_start = max(ow * stride - padding, 0);
    const int iw_end = min(ow * stride - padding + kernel_size * dilation, input_width);

    // Base offset for current batch and channel
    const int base_offset = b * (channels * input_height * input_width) +
                           c * (input_height * input_width);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Iterate only over valid positions
    for (int ih = ih_start; ih < ih_end; ih += dilation) {
        const int row_offset = ih * input_width;
        #pragma unroll 4
        for (int iw = iw_start; iw < iw_end; iw += dilation) {
            const int input_idx = base_offset + row_offset + iw;
            max_val = max(max_val, input[input_idx]);
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

    // Use multiple of warp size for thread block size
    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}