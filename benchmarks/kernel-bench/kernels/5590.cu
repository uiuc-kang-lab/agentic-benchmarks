#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int stride,
    int padding,
    int dilation
) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = blockIdx.z;
    
    if (ow >= output_width || oh >= output_height || bc >= batch_channels) return;

    const int b = bc / gridDim.z;
    const int c = bc % gridDim.z;

    const int input_batch_offset = b * input_height * input_width;
    const int base_h = oh * stride - padding;
    const int base_w = ow * stride - padding;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    #pragma unroll
    for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
        const int ih = base_h + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                const int iw = base_w + kw * dilation;
                if (iw >= 0 && iw < input_width) {
                    max_val = max(max_val, __ldg(&input[input_batch_offset + c * (input_height * input_width) + ih * input_width + iw]));
                }
            }
        }
    }

    output[bc * output_height * output_width + oh * output_width + ow] = max_val;
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

    dim3 block(16, 16);
    dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_forward", ([&] {
        switch(kernel_size) {
            case 3:
                max_pool2d_optimized<scalar_t, 3><<<grid, block>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size * channels,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                    stride,
                    padding,
                    dilation
                );
                break;
            default:
                max_pool2d_optimized<scalar_t, -1><<<grid, block>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size * channels,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                    stride,
                    padding,
                    dilation
                );
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D optimized forward (CUDA)");
}