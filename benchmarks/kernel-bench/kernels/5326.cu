#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Use constant memory for frequently accessed parameters
__constant__ int const_kernel_size;
__constant__ int const_stride;
__constant__ int const_padding;
__constant__ int const_dilation;

template <typename scalar_t>
__global__ void max_pool2d_kernel_constant(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    const int input_batch_offset = b * (channels * input_height * input_width);
    const int input_channel_offset = c * (input_height * input_width);
    const int input_idx_base = input_batch_offset + input_channel_offset;

    #pragma unroll
    for (int kh = 0; kh < const_kernel_size; kh++) {
        const int ih = oh * const_stride - const_padding + kh * const_dilation;
        if (ih >= 0 && ih < input_height) {
            const int ih_offset = ih * input_width;

            #pragma unroll
            for (int kw = 0; kw < const_kernel_size; kw++) {
                const int iw = ow * const_stride - const_padding + kw * const_dilation;
                if (iw >= 0 && iw < input_width) {
                    const int input_idx = input_idx_base + ih_offset + iw;
                    max_val = max(max_val, __ldg(&input[input_idx]));
                }
            }
        }
    }

    output[output_idx] = max_val;
}

void set_constant_memory(int kernel_size, int stride, int padding, int dilation) {
    cudaMemcpyToSymbol(const_kernel_size, &kernel_size, sizeof(int));
    cudaMemcpyToSymbol(const_stride, &stride, sizeof(int));
    cudaMemcpyToSymbol(const_padding, &padding, sizeof(int));
    cudaMemcpyToSymbol(const_dilation, &dilation, sizeof(int));
}

torch::Tensor max_pool2d_cuda_forward_constant(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    set_constant_memory(kernel_size, stride, padding, dilation);

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward_constant", ([&] {
        max_pool2d_kernel_constant<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward_constant, "Max Pool 2D forward with constant memory (CUDA)");
}