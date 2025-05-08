#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_optim_combined_kernel(
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
    // 2D spatial coordinates with 3D block-strided batch/channel loop
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc_step = gridDim.z * blockDim.z;
    const int num_bc = batch_size * channels;

    for (int bc_idx = blockIdx.z * blockDim.z + threadIdx.z;
         bc_idx < num_bc;
         bc_idx += bc_step) {
        
        if (ow >= output_width || oh >= output_height) continue;

        const int b = bc_idx / channels;
        const int c = bc_idx % channels;
        const int input_offset = (b * channels + c) * input_height * input_width;

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        if (KERNEL_SIZE > 0) {
            #pragma unroll
            for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                int ih = oh * stride - padding + kh * dilation;
                if (ih >= 0 && ih < input_height) {
                    #pragma unroll
                    for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                        int iw = ow * stride - padding + kw * dilation;
                        if (iw >= 0 && iw < input_width) {
                            max_val = fmaxf(max_val, __ldg(&input[input_offset + ih * input_width + iw]));
                        }
                    }
                }
            }
        } else {
            for (int kh = 0; kh < kernel_size; kh++) {
                int ih = oh * stride - padding + kh * dilation;
                if (ih >= 0 && ih < input_height) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int iw = ow * stride - padding + kw * dilation;
                        if (iw >= 0 && iw < input_width) {
                            max_val = fmaxf(max_val, input[input_offset + ih * input_width + iw]);
                        }
                    }
                }
            }
        }

        output[((b * channels + c) * output_height + oh) * output_width + ow] = max_val;
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
    const int num_bc = batch_size * channels;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const dim3 block(32, 8, 4);  // Optimized 2D spatial block + parallel bc dim
    const dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        (num_bc + block.z - 1) / block.z
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        switch(kernel_size) {
            case 2:
                max_pool2d_optim_combined_kernel<scalar_t, 2><<<grid, block>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, channels,
                    input_height, input_width,
                    output_height, output_width,
                    kernel_size, stride, padding, dilation
                );
                break;
            case 3:
                max_pool2d_optim_combined_kernel<scalar_t, 3><<<grid, block>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, channels,
                    input_height, input_width,
                    output_height, output_width,
                    kernel_size, stride, padding, dilation
                );
                break;
            default:
                max_pool2d_optim_combined_kernel<scalar_t, -1><<<grid, block>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, channels,
                    input_height, input_width,
                    output_height, output_width,
                    kernel_size, stride, padding, dilation
                );
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Optimized Combined Max Pool 2D (CUDA)");
}
