#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_hybrid_kernel(
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
    const int total_elements
) {
    if (output_height <= 32 && output_width <= 32) {
        const int ow = blockIdx.x * blockDim.x + threadIdx.x;
        const int oh = blockIdx.y * blockDim.y + threadIdx.y;
        const int c = blockIdx.z % channels;
        const int b = blockIdx.z / channels;

        if (b >= batch_size || c >= channels || oh >= output_height || ow >= output_width) return;

        const int input_base = b * channels * input_height * input_width + 
                              c * input_height * input_width;

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        #pragma unroll
        for (int kh = 0; kh < KERNEL_SIZE; kh++) {
            const int ih = oh * stride - padding + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                #pragma unroll
                for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                    const int iw = ow * stride - padding + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        max_val = fmaxf(max_val, __ldg(&input[input_base + ih * input_width + iw]));
                    }
                }
            }
        }

        output[((b * channels + c) * output_height + oh) * output_width + ow] = max_val;
    }
    else {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int gridSize = blockDim.x * gridDim.x;
        
        while (index < total_elements) {
            const int ow = index % output_width;
            const int oh = (index / output_width) % output_height;
            const int c  = (index / (output_width * output_height)) % channels;
            const int b  = index / (output_width * output_height * channels);

            scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
            const int input_base = b * channels * input_height * input_width + 
                                 c * input_height * input_width;

            #pragma unroll
            for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                const int ih = oh * stride - padding + kh * dilation;
                if (ih >= 0 && ih < input_height) {
                    #pragma unroll
                    for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                        const int iw = ow * stride - padding + kw * dilation;
                        if (iw >= 0 && iw < input_width) {
                            max_val = fmaxf(max_val, __ldg(&input[input_base + ih * input_width + iw]));
                        }
                    }
                }
            }

            output[index] = max_val;
            index += gridSize;
        }
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

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int total_elements = batch_size * channels * output_height * output_width;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    dim3 block, grid;
    if (output_height <= 32 && output_width <= 32) {
        block = dim3(32, 8);
        grid = dim3(
            (output_width + block.x - 1) / block.x,
            (output_height + block.y - 1) / block.y,
            batch_size * channels
        );
    } else {
        block = dim3(256);
        grid = dim3((total_elements + block.x - 1) / block.x);
    }

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        switch(kernel_size) {
            case 2:
                max_pool2d_hybrid_kernel<scalar_t, 2><<<grid, block>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    channels,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                    stride,
                    padding,
                    dilation,
                    total_elements
                );
                break;
            case 3:
                max_pool2d_hybrid_kernel<scalar_t, 3><<<grid, block>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    channels,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                    stride,
                    padding,
                    dilation,
                    total_elements
                );
                break;
            default:
                max_pool2d_hybrid_kernel<scalar_t, -1><<<grid, block>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    channels,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                    stride,
                    padding,
                    dilation,
                    total_elements
                );
        }
    }));

    return output;
}