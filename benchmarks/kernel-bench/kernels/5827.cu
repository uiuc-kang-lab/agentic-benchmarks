#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
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
    const int total_output = batch_size * channels * output_height * output_width;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;

    if (warp_id >= total_output) return;

    // Decode output coordinates
    const int ow = warp_id % output_width;
    const int oh = (warp_id / output_width) % output_height;
    const int c = (warp_id / (output_width * output_height)) % channels;
    const int b = warp_id / (output_width * output_height * channels);

    // Each thread handles a (kh, kw) position
    const int kh = lane_id / kernel_size;
    const int kw = lane_id % kernel_size;
    scalar_t val = -std::numeric_limits<scalar_t>::infinity();

    if (kh < kernel_size && kw < kernel_size) {
        const int ih = oh * stride - padding + kh * dilation;
        const int iw = ow * stride - padding + kw * dilation;

        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
            const int input_idx = b * (channels * input_height * input_width) +
                                c * (input_height * input_width) +
                                ih * input_width + iw;
            val = input[input_idx];
        }
    }

    // Warp-level max reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        scalar_t tmp = __shfl_down_sync(0xffffffff, val, offset);
        val = max(val, tmp);
    }

    if (lane_id == 0) {
        output[warp_id] = val;
    }
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

    const int total_output = batch_size * channels * output_height * output_width;
    const int threads_per_block = 256;
    const int blocks = (total_output * 32 + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with warp shuffle reduction");
}