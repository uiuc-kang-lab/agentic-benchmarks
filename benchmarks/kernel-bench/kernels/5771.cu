#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_kernel_optimized(
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
    const int TILE_SIZE = 32;
    __shared__ scalar_t shared_tile[TILE_SIZE][TILE_SIZE];

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int output_idx = blockIdx.x * blockDim.x * blockDim.y + tid;

    if (output_idx >= batch_size * channels * output_height * output_width) return;

    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Load relevant input data into shared memory
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    for (int ph = 0; ph < kernel_size; ph += TILE_SIZE) {
        for (int pw = 0; pw < kernel_size; pw += TILE_SIZE) {
            // Load tile into shared memory
            const int ih = ih_start + (ph + ty) * dilation;
            const int iw = iw_start + (pw + tx) * dilation;
            
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                shared_tile[ty][tx] = input[b * (channels * input_height * input_width) +
                                            c * (input_height * input_width) +
                                            ih * input_width + iw];
            } else {
                shared_tile[ty][tx] = -std::numeric_limits<scalar_t>::infinity();
            }
            __syncthreads();

            // Process the tile
            for (int k = 0; k < min(TILE_SIZE, kernel_size - ph); ++k) {
                for (int l = 0; l < min(TILE_SIZE, kernel_size - pw); ++l) {
                    if (ty + k < TILE_SIZE && tx + l < TILE_SIZE) {
                        max_val = max(max_val, shared_tile[ty + k][tx + l]);
                    }
                }
            }
            __syncthreads();
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

    const dim3 threads(16, 16);
    const dim3 blocks(
        (batch_size * channels * output_height * output_width + threads.x * threads.y - 1) / (threads.x * threads.y)
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel_optimized<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with shared memory (CUDA)");
}