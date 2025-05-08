#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Shared memory tile size
#define TILE_SIZE 16

template <typename scalar_t>
__global__ void optimized_max_pool2d_tiled_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* output,
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
    __shared__ scalar_t shared_input[TILE_SIZE][TILE_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    // Calculate batch and channel indices
    const int b = bz / channels;
    const int c = bz % channels;
    
    // Calculate output coordinates
    const int oh_start = by * TILE_SIZE;
    const int ow_start = bx * TILE_SIZE;
    const int oh = oh_start + ty;
    const int ow = ow_start + tx;
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    if (oh < output_height && ow < output_width) {
        // Process kernel window with tiled approach
        for (int kh = 0; kh < kernel_size; kh += TILE_SIZE) {
            for (int kw = 0; kw < kernel_size; kw += TILE_SIZE) {
                // Load tile into shared memory
                const int ih_base = oh * stride - padding + kh * dilation;
                const int iw_base = ow * stride - padding + kw * dilation;
                
                if (ih_base + ty >= 0 && ih_base + ty < input_height &&
                    iw_base + tx >= 0 && iw_base + tx < input_width) {
                    const int input_idx = b * (channels * input_height * input_width) +
                                        c * (input_height * input_width) +
                                        (ih_base + ty) * input_width +
                                        (iw_base + tx);
                    shared_input[ty][tx] = __ldg(&input[input_idx]);
                } else {
                    shared_input[ty][tx] = -std::numeric_limits<scalar_t>::infinity();
                }
                
                __syncthreads();
                
                // Process the tile
                for (int i = 0; i < min(TILE_SIZE, kernel_size - kh); i++) {
                    for (int j = 0; j < min(TILE_SIZE, kernel_size - kw); j++) {
                        const int ih = oh * stride - padding + (kh + i) * dilation;
                        const int iw = ow * stride - padding + (kw + j) * dilation;
                        
                        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                            max_val = max(max_val, shared_input[i][j]);
                        }
                    }
                }
                
                __syncthreads();
            }
        }
        
        // Write output
        if (oh < output_height && ow < output_width) {
            const int output_idx = b * (channels * output_height * output_width) +
                                 c * (output_height * output_width) +
                                 oh * output_width +
                                 ow;
            output[output_idx] = max_val;
        }
    }
}

torch::Tensor optimized_max_pool2d_cuda_forward(
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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (output_width + TILE_SIZE - 1) / TILE_SIZE,
        (output_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_max_pool2d_cuda_forward", ([&] {
        optimized_max_pool2d_tiled_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &optimized_max_pool2d_cuda_forward, "Optimized Max Pool 2D forward (CUDA)");
}