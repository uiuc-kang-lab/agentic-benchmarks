#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// This kernel uses 2D thread blocks for the spatial dimensions of the output and a 3D grid (using the z-dimension) to cover batch and channel dimensions.
// This mapping reduces the computational overhead for index calculations by directly assigning threads to output spatial positions
// and improves memory access patterns, leading to potentially better runtime performance.

template <typename scalar_t>
__global__ void max_pool2d_kernel_2d_index(
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
    // blockIdx.z indexes over (batch_size * channels)
    int bc = blockIdx.z;
    int b = bc / channels;
    int c = bc % channels;

    // Compute output coordinates using 2D thread blocks
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;

    if (ow >= output_width || oh >= output_height) return;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    // Pre-calculate the offset for this (batch, channel)
    int input_channel_offset = b * (channels * input_height * input_width) + c * (input_height * input_width);

    // Loop over the kernel window
    for (int kh = 0; kh < kernel_size; kh++) {
        int ih = oh * stride - padding + kh * dilation;
        if (ih < 0 || ih >= input_height) continue;
        int ih_offset = ih * input_width;
        for (int kw = 0; kw < kernel_size; kw++) {
            int iw = ow * stride - padding + kw * dilation;
            if (iw < 0 || iw >= input_width) continue;
            int input_idx = input_channel_offset + ih_offset + iw;
            max_val = max(max_val, __ldg(&input[input_idx]));
        }
    }

    // Compute the output index and store the result
    int out_idx = b * (channels * output_height * output_width) + c * (output_height * output_width) + oh * output_width + ow;
    output[out_idx] = max_val;
}


torch::Tensor max_pool2d_cuda_forward(
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

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Define 2D block dimensions for the spatial domain
    const int TILE_WIDTH = 16;
    const int TILE_HEIGHT = 16;
    dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);

    // Grid dimensions: x for width, y for height, z for batch * channels
    dim3 gridDim((output_width + TILE_WIDTH - 1) / TILE_WIDTH,
                 (output_height + TILE_HEIGHT - 1) / TILE_HEIGHT,
                 batch_size * channels);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel_2d_index<scalar_t><<<gridDim, blockDim>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, channels,
            input_height, input_width,
            output_height, output_width,
            kernel_size, stride, padding, dilation);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with 2D thread/block indexing (CUDA)");
}
