#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int KERNEL_SIZE>
__device__ void compute_tiled_max(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int input_offset,
    const int output_offset,
    const int input_height,
    const int input_width,
    const int output_width,
    const int stride,
    const int padding,
    const int dilation
) {
    const int ow_start = threadIdx.x * 4;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;

    #pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        const int ow = ow_start + tile;
        if (ow >= output_width) return;

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        const int base_h = oh * stride - padding;
        const int base_w = ow * stride - padding;

        #pragma unroll
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            const int ih = base_h + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                #pragma unroll
                for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                    const int iw = base_w + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        max_val = fmaxf(max_val, __ldg(&input[input_offset + ih * input_width + iw]));
                    }
                }
            }
        }
        output[output_offset + oh * output_width + ow] = max_val;
    }
}

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_tiled_kernel(
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
    const int dilation
) {
    const int bc = blockIdx.z;
    const int b = bc / channels;
    const int c = bc % channels;
    
    if (b >= batch_size) return;

    const int input_offset = (b * channels + c) * input_height * input_width;
    const int output_offset = (b * channels + c) * output_height * output_width;

    compute_tiled_max<scalar_t, KERNEL_SIZE>(
        input, output, input_offset, output_offset,
        input_height, input_width, output_width,
        stride, padding, dilation
    );
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

    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const dim3 block(32, 8);  // 256 threads total (32*8), 4 elements per thread in x-dim
    const dim3 grid(
        (output_width + (block.x * 4) - 1) / (block.x * 4),
        (output_height + block.y - 1) / block.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_forward", ([&] {
        switch(kernel_size) {
            case 2:
                max_pool2d_tiled_kernel<scalar_t, 2><<<grid, block>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, channels,
                    input_height, input_width,
                    output_height, output_width,
                    stride, padding, dilation
                );
                break;
            case 3:
                max_pool2d_tiled_kernel<scalar_t, 3><<<grid, block>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, channels,
                    input_height, input_width,
                    output_height, output_width,
                    stride, padding, dilation
                );
                break;
            case 4:
                max_pool2d_tiled_kernel<scalar_t, 4><<<grid, block>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, channels,
                    input_height, input_width,
                    output_height, output_width,
                    stride, padding, dilation
                );
                break;
            default:
                max_pool2d_tiled_kernel<scalar_t, -1><<<grid, block>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, channels,
                    input_height, input_width,
                    output_height, output_width,
                    stride, padding, dilation
                );
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Tiled Max Pool 2D with thread coarsening (CUDA)");
}
