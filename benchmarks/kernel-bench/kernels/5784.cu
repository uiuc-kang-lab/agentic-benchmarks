#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimize workload balance by adjusting grid and block sizes

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
    const int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (ow < output_width && oh < output_height && c < channels * batch_size) {
        const int b = c / channels;
        const int channel_idx = c % channels;

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int ih = oh * stride - padding + kh * dilation;
                const int iw = ow * stride - padding + kw * dilation;

                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    const int input_idx = (b * channels + channel_idx) * (input_height * input_width) +
                                        ih * input_width + iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }

        const int output_idx = (b * channels + channel_idx) * (output_height * output_width) +
                               oh * output_width + ow;
        output[output_idx] = max_val;
    }
}

torch::Tensor max_pool2d_cuda_forward_balanced(
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

    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const dim3 threads(16, 16, 1); // each thread processes a part of the output
    const dim3 blocks((output_width + threads.x - 1) / threads.x,
                      (output_height + threads.y - 1) / threads.y,
                      ((batch_size * channels) + threads.z - 1) / threads.z);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_forward_balanced", ([&] {
        max_pool2d_balanced_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward_balanced, "Max Pool 2D forward with workload balancing (CUDA)");
}