#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int constant_kernel_size;
__constant__ int constant_stride;
__constant__ int constant_padding;
__constant__ int constant_dilation;

// Max Pool 2D using constant memory
template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* input,
    scalar_t* output,
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

    #pragma unroll
    for (int kh = 0; kh < constant_kernel_size; kh++) {
        #pragma unroll
        for (int kw = 0; kw < constant_kernel_size; kw++) {
            const int ih = oh * constant_stride - constant_padding + kh * constant_dilation;
            const int iw = ow * constant_stride - constant_padding + kw * constant_dilation;

            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                const int input_idx = b * (channels * input_height * input_width) +
                                    c * (input_height * input_width) +
                                    ih * input_width +
                                    iw;
                max_val = max(max_val, input[input_idx]);
            }
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

    int h_kernel_size = kernel_size;
    int h_stride = stride;
    int h_padding = padding;
    int h_dilation = dilation;

    cudaMemcpyToSymbol(constant_kernel_size, &h_kernel_size, sizeof(int));
    cudaMemcpyToSymbol(constant_stride, &h_stride, sizeof(int));
    cudaMemcpyToSymbol(constant_padding, &h_padding, sizeof(int));
    cudaMemcpyToSymbol(constant_dilation, &h_dilation, sizeof(int));

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
            output_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}