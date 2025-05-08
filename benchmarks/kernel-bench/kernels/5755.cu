#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

template <typename scalar_t>
__global__ void max_pool2d_kernel_unroll(
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

    // Compute output indices
    int ow = index % output_width;
    int tmp = index / output_width;
    int oh = tmp % output_height;
    tmp /= output_height;
    int c = tmp % channels;
    int b = tmp / channels;

    // Initialize max value
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Loop over kernel window with unrolling
    #pragma unroll
    for (int kh = 0; kh < kernel_size; ++kh) {
        int ih = oh * stride - padding + kh * dilation;
        #pragma unroll
        for (int kw = 0; kw < kernel_size; ++kw) {
            int iw = ow * stride - padding + kw * dilation;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                int input_idx = ((b * channels + c) * input_height + ih) * input_width + iw;
                max_val = max(max_val, input[input_idx]);
            }
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
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    // Compute output dimensions (explicitly cast multiplications to avoid narrowing warnings)
    const int output_height = static_cast<int>(((static_cast<int64_t>(input_height) + 2LL * padding - dilation * (kernel_size - 1) - 1) / stride) + 1);
    const int output_width  = static_cast<int>(((static_cast<int64_t>(input_width) + 2LL * padding - dilation * (kernel_size - 1) - 1) / stride) + 1);

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int total_elements = batch_size * channels * output_height * output_width;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel_unroll<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with loop unrolling (CUDA)");
}
