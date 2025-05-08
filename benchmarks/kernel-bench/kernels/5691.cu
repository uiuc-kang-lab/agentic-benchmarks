#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t, int KERNEL_SIZE, int DILATION>
__global__ void max_pool2d_tuned_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding
) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;

    if (ow >= output_width || oh >= output_height || b >= batch_size) return;

    const int base_input_h = oh * stride - padding;
    const int base_input_w = ow * stride - padding;
    const int input_offset = (b * channels + c) * input_height * input_width;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    #pragma unroll
    for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
        const int ih = base_input_h + kh * DILATION;
        if (ih >= 0 && ih < input_height) {
            const int row_offset = ih * input_width;
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                const int iw = base_input_w + kw * DILATION;
                if (iw >= 0 && iw < input_width) {
                    const int idx = input_offset + row_offset + iw;
                    if constexpr (DILATION == 1) {  // Non-dilated path
                        max_val = fmaxf(max_val, input[idx]);
                    } else {                       // General case with __ldg
                        max_val = fmaxf(max_val, __ldg(&input[idx]));
                    }
                }
            }
        }
    }

    output[((b * channels + c) * output_height + oh) * output_width + ow] = max_val;
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

    // Optimized block size for coalesced writes and H100 occupancy
    const dim3 block(32, 4); // 128 threads per block for better occupancy control
    const dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_forward", [&] {
        if (dilation == 1) { // Specialize for common dilation=1 case
            switch(kernel_size) {
                case 2:
                    max_pool2d_tuned_kernel<scalar_t, 2, 1><<<grid, block>>>(
                        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                        batch_size, channels, input_height, input_width,
                        output_height, output_width, stride, padding
                    );
                    break;
                case 3:
                    max_pool2d_tuned_kernel<scalar_t, 3, 1><<<grid, block>>>(
                        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                        batch_size, channels, input_height, input_width,
                        output_height, output_width, stride, padding
                    );
                    break;
                default:
                    max_pool2d_tuned_kernel<scalar_t, -1, 1><<<grid, block>>>(
                        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                        batch_size, channels, input_height, input_width,
                        output_height, output_width, stride, padding
                    );
            }
        } else { // General case with arbitrary dilation
            switch(kernel_size) {
                case 2:
                    max_pool2d_tuned_kernel<scalar_t, 2, -1><<<grid, block>>>(
                        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                        batch_size, channels, input_height, input_width,
                        output_height, output_width, stride, padding
                    );
                    break;
                case 3:
                    max_pool2d_tuned_kernel<scalar_t, 3, -1><<<grid, block>>>(
                        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                        batch_size, channels, input_height, input_width,
                        output_height, output_width, stride, padding
                    );
                    break;
                default:
                    max_pool2d_tuned_kernel<scalar_t, -1, -1><<<grid, block>>>(
                        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                        batch_size, channels, input_height, input_width,
                        output_height, output_width, stride, padding
                    );
            }
        }
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "MaxPool2D tuned with coalescing");
}
