#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

// This kernel uses 2D thread blocks for the spatial (output height & width) dimensions
// and uses the third grid dimension to map the batch and channel indices.

template <typename scalar_t>
__global__ void max_pool2d_kernel_2d(
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
    // Compute spatial output coordinates using 2D block and thread indexing
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Use the third grid dimension to cover the batch and channel dimensions
    const int bc = blockIdx.z;  // bc = b * channels + c
    if (ow >= output_width || oh >= output_height) return;

    const int c = bc % channels;
    const int b = bc / channels;

    // Initialize max value to negative infinity
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Loop over the kernel window
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = oh * stride - padding + kh * dilation;
            int iw = ow * stride - padding + kw * dilation;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                int input_idx = b * (channels * input_height * input_width) +
                                c * (input_height * input_width) +
                                ih * input_width + iw;
                scalar_t val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }

    int output_idx = b * (channels * output_height * output_width) +
                     c * (output_height * output_width) +
                     oh * output_width + ow;
    output[output_idx] = max_val;
}

// Host function to launch the kernel
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

    // Use 16x16 thread blocks to cover the output spatial dimensions
    const dim3 block(16, 16);
    // Grid dimensions: x covers output_width, y covers output_height, and z covers batch_size * channels
    const dim3 grid((output_width + block.x - 1) / block.x,
                    (output_height + block.y - 1) / block.y,
                    batch_size * channels);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel_2d<scalar_t><<<grid, block>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Optimized Max Pool 2D forward (CUDA) with 2D indexing");
}
