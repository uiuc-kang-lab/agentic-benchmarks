#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

template <typename scalar_t>
__global__ void max_pool2d_combined_kernel(
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
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = blockIdx.z;
    const int b = bc / channels;
    const int c = bc % channels;

    if (b >= batch_size || c >= channels || oh >= output_height || ow >= output_width)
        return;

    const scalar_t* input_channel = input + (b * channels + c) * input_height * input_width;
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Precompute valid ranges
    int base_h = oh * stride - padding;
    int base_w = ow * stride - padding;

    int kh_start = 0;
    if (base_h < 0)
        kh_start = (-base_h + dilation - 1) / dilation;
    int kh_end = kernel_size;
    if (base_h + (kernel_size - 1) * dilation >= input_height)
        kh_end = min((input_height - base_h + dilation - 1) / dilation, kernel_size);

    int kw_start = 0;
    if (base_w < 0)
        kw_start = (-base_w + dilation - 1) / dilation;
    int kw_end = kernel_size;
    if (base_w + (kernel_size - 1) * dilation >= input_width)
        kw_end = min((input_width - base_w + dilation - 1) / dilation, kernel_size);

    // Handle common kernel sizes with unrolling
    if (kernel_size == 2) {
        #pragma unroll
        for (int kh = kh_start; kh < kh_end; ++kh) {
            #pragma unroll
            for (int kw = kw_start; kw < kw_end; ++kw) {
                const int ih = base_h + kh * dilation;
                const int iw = base_w + kw * dilation;
                max_val = max(max_val, input_channel[ih * input_width + iw]);
            }
        }
    } else if (kernel_size == 3) {
        #pragma unroll
        for (int kh = kh_start; kh < kh_end; ++kh) {
            #pragma unroll
            for (int kw = kw_start; kw < kw_end; ++kw) {
                const int ih = base_h + kh * dilation;
                const int iw = base_w + kw * dilation;
                max_val = max(max_val, input_channel[ih * input_width + iw]);
            }
        }
    } else {
        for (int kh = kh_start; kh < kh_end; ++kh) {
            for (int kw = kw_start; kw < kw_end; ++kw) {
                const int ih = base_h + kh * dilation;
                const int iw = base_w + kw * dilation;
                max_val = max(max_val, input_channel[ih * input_width + iw]);
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

    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const dim3 threads(32, 8);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_combined_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA) combined");
}