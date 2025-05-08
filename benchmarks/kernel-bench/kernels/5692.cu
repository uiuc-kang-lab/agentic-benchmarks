#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// This kernel uses a grid-stride loop with a tunable block size. Use the BLOCK_SIZE macro (e.g., 32, 64, 128, 256, 512) to experiment with optimal configuration on your hardware.

template <typename scalar_t>
__global__ void max_pool2d_tunable_blocksize_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int total_elements,
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gridSize = blockDim.x * gridDim.x;

    while (idx < total_elements) {
        // Decode flat index into (b, c, oh, ow)
        int ow = idx % output_width;
        int oh = (idx / output_width) % output_height;
        int c  = (idx / (output_width * output_height)) % channels;
        int b  = idx / (output_width * output_height * channels);

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        int input_base = (b * channels + c) * input_height * input_width;

        // Iterate over the pooling window
        for (int kh = 0; kh < kernel_size; ++kh) {
            int ih = oh * stride - padding + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int iw = ow * stride - padding + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        int input_index = input_base + ih * input_width + iw;
                        max_val = fmaxf(max_val, __ldg(&input[input_index]));
                    }
                }
            }
        }
        
        output[idx] = max_val;
        idx += gridSize;
    }
}

// Host function: computes output dimensions and dispatches the kernel with a tunable block size defined by the BLOCK_SIZE macro.

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
    const int output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());
    
    int total_elements = batch_size * channels * output_height * output_width;
    int threads = BLOCK_SIZE;
    int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_forward", ([&] {
        max_pool2d_tunable_blocksize_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total_elements,
            batch_size, channels,
            input_height, input_width,
            output_height, output_width,
            kernel_size, stride, padding, dilation
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with tunable block size (CUDA)");
}
