#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int const_params[5]; // Hold kernel_size, stride, padding, dilation, in terms of setup

// Update the kernel to use constant memory for fixed parameters
template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    // Retrieve fixed parameters from constant memory
    const int kernel_size = const_params[0];
    const int stride = const_params[1];
    const int pad = const_params[2];
    const int dilation = const_params[3];

    // Determine global output coordinates
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;

    // Compute the starting coordinates for the output block
    const int out_start_x = blockIdx.x * blockDim.x;
    const int out_start_y = blockIdx.y * blockDim.y;

    // Compute the corresponding starting input coordinate for the shared tile
    const int in_tile_start_x = out_start_x * stride - pad;
    const int in_tile_start_y = out_start_y * stride - pad;

    // Determine shared memory tile dimensions
    const int tile_width = blockDim.x * stride + (kernel_size - 1) * dilation;
    const int tile_height = blockDim.y * stride + (kernel_size - 1) * dilation;

    // Allocate shared memory tile
    extern __shared__ scalar_t shared_tile[];

    // Each thread loads one or more elements into shared memory
    for (int ty = threadIdx.y; ty < tile_height; ty += blockDim.y) {
        for (int tx = threadIdx.x; tx < tile_width; tx += blockDim.x) {
            int ix = in_tile_start_x + tx;
            int iy = in_tile_start_y + ty;
            int shared_idx = ty * tile_width + tx;
            if (ix >= 0 && ix < input_width && iy >= 0 && iy < input_height) {
                int input_idx = ((b * channels + c) * input_height + iy) * input_width + ix;
                shared_tile[shared_idx] = input[input_idx];
            } else {
                shared_tile[shared_idx] = -std::numeric_limits<scalar_t>::infinity();
            }
        }
    }
    __syncthreads();

    // Only process if the thread corresponds to a valid output pixel
    if (ox < output_width && oy < output_height) {
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        // Local starting position in the shared tile for this output pixel
        int local_x = threadIdx.x * stride;
        int local_y = threadIdx.y * stride;
        
        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int sx = local_x + kw * dilation;
                int sy = local_y + kh * dilation;
                int global_x = in_tile_start_x + sx;
                int global_y = in_tile_start_y + sy;
                if (global_x >= 0 && global_x < input_width && global_y >= 0 && global_y < input_height) {
                    int tile_idx = sy * tile_width + sx;
                    max_val = max(max_val, shared_tile[tile_idx]);
                }
            }
        }
        int output_idx = ((b * channels + c) * output_height + oy) * output_width + ox;
        output[output_idx] = max_val;
    }
}

// Host function to launch the CUDA kernel with setup of constant memory
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

    // Setup constant memory
    int h_const_params[5] = {kernel_size, stride, padding, dilation, 0};
    cudaMemcpyToSymbol(const_params, h_const_params, sizeof(int) * 4);

    // 2D block configuration
    dim3 threads(32, 8);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

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