#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_kernel_unroll_3x3(
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
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_height * output_width) return;

    const int ow = idx % output_width;
    const int oh = (idx / output_width) % output_height;
    const int c = (idx / (output_width * output_height)) % channels;
    const int b = idx / (output_width * output_height * channels);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Manual unroll for 3x3 kernel
    #define CHECK_AND_UPDATE(kh, kw) \
    { \
        const int ih = oh * stride - padding + (kh) * dilation; \
        const int iw = ow * stride - padding + (kw) * dilation; \
        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) { \
            const int input_idx = ((b * channels + c) * input_height + ih) * input_width + iw; \
            max_val = max(max_val, input[input_idx]); \
        } \
    }

    // Fully unrolled 3x3 kernel
    CHECK_AND_UPDATE(0, 0)
    CHECK_AND_UPDATE(0, 1)
    CHECK_AND_UPDATE(0, 2)
    CHECK_AND_UPDATE(1, 0)
    CHECK_AND_UPDATE(1, 1)
    CHECK_AND_UPDATE(1, 2)
    CHECK_AND_UPDATE(2, 0)
    CHECK_AND_UPDATE(2, 1)
    CHECK_AND_UPDATE(2, 2)

    #undef CHECK_AND_UPDATE

    output[idx] = max_val;
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

    const auto output_height = static_cast<int64_t>((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = static_cast<int64_t>((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int total_elements = batch_size * channels * output_height * output_width;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel_unroll_3x3<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            stride,
            padding,
            dilation
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with manual unrolling (CUDA)");
}