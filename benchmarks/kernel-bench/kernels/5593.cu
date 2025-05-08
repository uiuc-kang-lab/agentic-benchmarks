#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_balanced_kernel(
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
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;

    if (ow >= output_width || oh >= output_height || b >= batch_size) return;

    const int input_batch_stride = channels * input_height * input_width;
    const int input_channel_stride = input_height * input_width;
    const int base_idx = b * input_batch_stride + c * input_channel_stride;
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;

    for (int kh = 0; kh < kernel_size; kh++) {
        const int ih = ih_start + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            const int row_idx = base_idx + ih * input_width;
            for (int kw = 0; kw < kernel_size; kw++) {
                const int iw = iw_start + kw * dilation;
                if (iw >= 0 && iw < input_width) {
                    max_val = max(max_val, __ldg(&input[row_idx + iw]));
                }
            }
        }
    }

    const int output_idx = b * (channels * output_height * output_width) +
                           c * (output_height * output_width) +
                           oh * output_width +
                           ow;
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

    const dim3 threads(16, 16);
    const dim3 blocks((output_width + threads.x - 1) / threads.x,
                      (output_height + threads.y - 1) / threads.y,
                      batch_size * channels);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_balanced_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, channels,
            input_height, input_width,
            output_height, output_width,
            kernel_size, stride, padding, dilation
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}