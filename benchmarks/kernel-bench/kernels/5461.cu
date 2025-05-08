#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimize thread and block indexing for better performance
template <typename scalar_t>
__global__ void max_pool2d_kernel_thread_block_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* output,
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
    // Use 2D grid and block for better spatial locality
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (ow >= output_width || oh >= output_height || c >= channels) return;

    for (int b = 0; b < batch_size; ++b) {
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                const int ih = oh * stride - padding + kh * dilation;
                const int iw = ow * stride - padding + kw * dilation;

                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    const int input_idx = b * (channels * input_height * input_width) +
                                        c * (input_height * input_width) +
                                        ih * input_width +
                                        iw;
                    max_val = max(max_val, __ldg(&input[input_idx]));
                }
            }
        }

        const int output_idx = b * (channels * output_height * output_width) +
                               c * (output_height * output_width) +
                               oh * output_width +
                               ow;
        output[output_idx] = max_val;
    }
}

torch::Tensor max_pool2d_cuda_forward_thread_block_optimized(
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
                      channels);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward_thread_block_optimized", ([&] {
        max_pool2d_kernel_thread_block_optimized<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward_thread_block_optimized, "Max Pool 2D forward with optimized thread and block indexing (CUDA)");
}