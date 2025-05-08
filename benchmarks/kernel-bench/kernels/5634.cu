#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t, int TILE_SIZE=8, int BLOCK_SIZE=256>
__global__ void max_pool2d_kernel_tiled(
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
    // Calculate tile position
    const int tile_x = blockIdx.x * TILE_SIZE;
    const int tile_y = blockIdx.y * TILE_SIZE;
    const int channel = blockIdx.z % channels;
    const int batch = blockIdx.z / channels;
    
    // Thread position within tile
    const int tx = threadIdx.x % TILE_SIZE;
    const int ty = threadIdx.x / TILE_SIZE;
    
    if (batch >= batch_size) return;
    
    const int oh = tile_y + ty;
    const int ow = tile_x + tx;
    
    if (oh < output_height && ow < output_width) {
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            const int ih = oh * stride - padding + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                #pragma unroll
                for (int kw = 0; kw < kernel_size; kw++) {
                    const int iw = ow * stride - padding + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        const int input_idx = batch * (channels * input_height * input_width) +
                                            channel * (input_height * input_width) +
                                            ih * input_width + iw;
                        max_val = max(max_val, input[input_idx]);
                    }
                }
            }
        }
        
        const int output_idx = batch * (channels * output_height * output_width) +
                              channel * (output_height * output_width) +
                              oh * output_width + ow;
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

    constexpr int TILE_SIZE = 8;
    constexpr int BLOCK_SIZE = 256;
    
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(
        (output_width + TILE_SIZE - 1) / TILE_SIZE,
        (output_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel_tiled<scalar_t, TILE_SIZE, BLOCK_SIZE><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}