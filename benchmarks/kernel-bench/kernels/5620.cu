#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Device function to compute max pooling
template <typename scalar_t>
__device__ scalar_t compute_max_pool(
    const scalar_t* __restrict__ input,
    const int input_offset,
    const int input_width,
    const int input_height,
    const int ih_start,
    const int iw_start,
    const int kernel_size,
    const int dilation) {

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    for (int kh = 0; kh < kernel_size; ++kh) {
        const int ih = ih_start + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int iw = iw_start + kw * dilation;
                if (iw >= 0 && iw < input_width) {
                    max_val = max(max_val, __ldg(&input[input_offset + ih * input_width + iw]));
                }
            }
        }
    }
    return max_val;
}

// Kernel function for max pooling
template <typename scalar_t>
__global__ void modularized_max_pool2d_kernel(
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
    const int dilation,
    const int kernel_size) {

    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = blockIdx.z;

    if (ow >= output_width || oh >= output_height) return;

    const int b = bc / channels;
    const int c = bc % channels;

    const int input_offset = b * channels * input_height * input_width 
                           + c * input_height * input_width;
    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;

    const scalar_t max_val = compute_max_pool<scalar_t>(
        input, input_offset, input_width, input_height, ih_start, iw_start, kernel_size, dilation);

    output[bc * output_height * output_width + oh * output_width + ow] = max_val;
}

// Host function to launch the kernel
torch::Tensor max_pool2d_modularized_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    dim3 threads(16, 16);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "modularized_max_pool2d_forward", [&] {
        modularized_max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            batch_size, channels, input_height, input_width,
            output_height, output_width, stride, padding, dilation, kernel_size);
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_modularized_forward, "Modularized Max Pool 2D forward (CUDA)");
}