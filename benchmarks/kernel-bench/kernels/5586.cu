#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int const_params[4];

// Define tile dimensions
#define TILE_DIM_X 16
#define TILE_DIM_Y 16

template <typename scalar_t>
__global__ void max_pool2d_tiled_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    const int kernel_size = const_params[0];
    const int stride = const_params[1];
    const int padding = const_params[2];
    const int dilation = const_params[3];

    // Calculate tile indices
    const int tile_x = blockIdx.x * TILE_DIM_X + threadIdx.x;
    const int tile_y = blockIdx.y * TILE_DIM_Y + threadIdx.y;
    const int channel = blockIdx.z % channels;
    const int batch = blockIdx.z / channels;

    // Check if this thread should compute output
    if (batch < batch_size && tile_y < output_height && tile_x < output_width) {
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            const int ih = tile_y * stride - padding + kh * dilation;
            
            #pragma unroll
            for (int kw = 0; kw < kernel_size; kw++) {
                const int iw = tile_x * stride - padding + kw * dilation;

                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    const int input_idx = batch * (channels * input_height * input_width) +
                                        channel * (input_height * input_width) +
                                        ih * input_width +
                                        iw;
                    max_val = max(max_val, __ldg(&input[input_idx]));
                }
            }
        }

        // Write output
        const int output_idx = batch * (channels * output_height * output_width) +
                             channel * (output_height * output_width) +
                             tile_y * output_width +
                             tile_x;
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
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int params[4] = {kernel_size, stride, padding, dilation};
    cudaMemcpyToSymbol(const_params, params, sizeof(int) * 4);

    // Configure grid and block dimensions for 2D tiling
    dim3 threads(TILE_DIM_X, TILE_DIM_Y);
    dim3 blocks(
        (output_width + TILE_DIM_X - 1) / TILE_DIM_X,
        (output_height + TILE_DIM_Y - 1) / TILE_DIM_Y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_tiled_kernel<scalar_t><<<blocks, threads>>>(
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