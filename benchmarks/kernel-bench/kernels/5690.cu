#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_coalesced_unrolled_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int channels,
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
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;
    
    if (ow >= output_width || oh >= output_height || b >= batch_size) return;

    const int input_offset = (b * channels + c) * input_height * input_width;
    const int base_h = oh * stride - padding;
    const int base_w = ow * stride - padding;
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Compile-time unrolling for common kernel sizes
    if constexpr (KERNEL_SIZE > 0) {
        #pragma unroll
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            const int ih = base_h + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                #pragma unroll
                for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                    const int iw = base_w + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        max_val = fmaxf(max_val, __ldg(input + input_offset + ih * input_width + iw));
                    }
                }
            }
        }
    } else {
        // Dynamic kernel size fallback
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            const int ih = base_h + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                    const int iw = base_w + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        max_val = fmaxf(max_val, input[input_offset + ih * input_width + iw]);
                    }
                }
            }
        }
    }

    output[(b * channels + c) * output_height * output_width + oh * output_width + ow] = max_val;
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

    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const dim3 block(32, 8); // Optimal for H100 memory coalescing
    const dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_forward", ([&] {
        switch(kernel_size) {
            case 2:
                max_pool2d_coalesced_unrolled_kernel<scalar_t, 2><<<grid, block>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation
                );
                break;
            case 3:
                max_pool2d_coalesced_unrolled_kernel<scalar_t, 3><<<grid, block>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation
                );
                break;
            case 4:
                max_pool2d_coalesced_unrolled_kernel<scalar_t, 4><<<grid, block>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation
                );
                break;
            default:
                max_pool2d_coalesced_unrolled_kernel<scalar_t, -1><<<grid, block>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation
                );
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D with coalesced accesses and loop unrolling (CUDA)");
}