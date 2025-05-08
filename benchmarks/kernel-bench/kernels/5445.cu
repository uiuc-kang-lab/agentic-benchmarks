#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Device function to calculate the maximum value in the window
template <typename scalar_t>
__device__ scalar_t calculate_max(
    const scalar_t* __restrict__ input,
    const int batch,
    const int channel,
    const int input_height,
    const int input_width,
    const int ih_start,
    const int iw_start,
    const int kernel_size,
    const int dilation
) {
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int ih = ih_start + kh * dilation;
            int iw = iw_start + kw * dilation;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                int input_idx = batch * (channel * input_height * input_width) +
                                channel * (input_height * input_width) +
                                ih * input_width + iw;
                max_val = max(max_val, __ldg(&input[input_idx]));
            }
        }
    }
    return max_val;
}

// Kernel to execute max pooling with modular functions
template <typename scalar_t>
__global__ void max_pool2d_kernel_modular(
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
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;

    // Call device function to compute max
    scalar_t max_val = calculate_max(input, b, c, input_height, input_width, ih_start, iw_start, kernel_size, dilation);

    output[output_idx] = max_val;
}

torch::Tensor max_pool2d_cuda_forward_modular(
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

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward_modular", ([&] {
        max_pool2d_kernel_modular<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward_modular, "Max Pool 2D forward with modular CUDA (CUDA)");
}
