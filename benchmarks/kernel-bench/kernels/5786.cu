#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Define tile dimensions for the spatial output
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// This kernel uses a 3D grid: gridDim.z covers (batch * channels), while gridDim.x and gridDim.y cover the output spatial dimensions. 
// Each thread computes one output element, ensuring an even distribution of work across threads and blocks.

template <typename scalar_t>
__global__ void max_pool2d_kernel_grid3d(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch,
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
    // Use blockIdx.z to identify the (batch, channel) pair
    int bc = blockIdx.z;
    int b = bc / channels;
    int c = bc % channels;

    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_y < output_height && out_x < output_width) {
        int in_y_start = out_y * stride - padding;
        int in_x_start = out_x * stride - padding;
        
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        // For a common case of 2x2 pooling, unroll the loop to reduce overhead
        if (kernel_size == 2) {
            #pragma unroll
            for (int kh = 0; kh < 2; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < 2; ++kw) {
                    int in_y = in_y_start + kh * dilation;
                    int in_x = in_x_start + kw * dilation;
                    if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                        int input_idx = ((b * channels + c) * input_height + in_y) * input_width + in_x;
                        max_val = max(max_val, input[input_idx]);
                    }
                }
            }
        } else {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_y = in_y_start + kh * dilation;
                    int in_x = in_x_start + kw * dilation;
                    if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                        int input_idx = ((b * channels + c) * input_height + in_y) * input_width + in_x;
                        max_val = max(max_val, input[input_idx]);
                    }
                }
            }
        }
        
        int output_idx = ((b * channels + c) * output_height + out_y) * output_width + out_x;
        output[output_idx] = max_val;
    }
}

torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch, channels, output_height, output_width}, input.options());

    // Set up a 3D grid where gridDim.z = batch * channels, and gridDim.x/y cover the spatial dimensions
    dim3 threads(TILE_WIDTH, TILE_HEIGHT, 1);
    dim3 blocks(
        (output_width + TILE_WIDTH - 1) / TILE_WIDTH,
        (output_height + TILE_HEIGHT - 1) / TILE_HEIGHT,
        batch * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_forward", ([&] {
        max_pool2d_kernel_grid3d<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch,
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D optimized grid3d forward (CUDA)");
}
