#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_grid_stride_kernel(
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
    const int output_size = batch_size * channels * output_height * output_width;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < output_size; idx += grid_stride) {
        const int b = idx / (channels * output_height * output_width);
        const int c = (idx / (output_height * output_width)) % channels;
        const int oh = (idx / output_width) % output_height;
        const int ow = idx % output_width;

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        const scalar_t* input_channel = input + (b * channels + c) * input_height * input_width;

        const int base_ih = oh * stride - padding;
        const int base_iw = ow * stride - padding;

        if (kernel_size == 2) {
            #pragma unroll
            for (int kh = 0; kh < 2; kh++) {
                const int ih = base_ih + kh * dilation;
                const bool valid_h = ih >= 0 && ih < input_height;
                #pragma unroll
                for (int kw = 0; kw < 2; kw++) {
                    const int iw = base_iw + kw * dilation;
                    if (valid_h && iw >= 0 && iw < input_width)
                        max_val = max(max_val, input_channel[ih * input_width + iw]);
                }
            }
        } else if (kernel_size == 3) {
            #pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                const int ih = base_ih + kh * dilation;
                const bool valid_h = ih >= 0 && ih < input_height;
                #pragma unroll
                for (int kw = 0; kw < 3; kw++) {
                    const int iw = base_iw + kw * dilation;
                    if (valid_h && iw >= 0 && iw < input_width)
                        max_val = max(max_val, input_channel[ih * input_width + iw]);
                }
            }
        } else {
            #pragma unroll 4
            for (int kh = 0; kh < kernel_size; kh++) {
                const int ih = base_ih + kh * dilation;
                const bool valid_h = ih >= 0 && ih < input_height;
                #pragma unroll 4
                for (int kw = 0; kw < kernel_size; kw++) {
                    const int iw = base_iw + kw * dilation;
                    if (valid_h && iw >= 0 && iw < input_width)
                        max_val = max(max_val, input_channel[ih * input_width + iw]);
                }
            }
        }

        output[idx] = max_val;
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

    const int output_size = batch_size * channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (output_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_grid_stride_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA) [Grid Stride]");
}