#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void max_pool2d_kernel(
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
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;

    if (ow >= output_width || oh >= output_height || c >= channels || b >= batch_size) return;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const int in_y_base = oh * stride - padding;
    const int in_x_base = ow * stride - padding;

    // Precompute valid kernel ranges
    int kh_start = max(0, (-in_y_base + dilation - 1) / dilation);
    int kh_end = min(kernel_size, (input_height - in_y_base + dilation - 1) / dilation);
    int kw_start = max(0, (-in_x_base + dilation - 1) / dilation);
    int kw_end = min(kernel_size, (input_width - in_x_base + dilation - 1) / dilation);

    #pragma unroll
    for (int kh = kh_start; kh < kh_end; ++kh) {
        const int iy = in_y_base + kh * dilation;
        const int y_offset = (b * channels + c) * input_height * input_width + iy * input_width;
        #pragma unroll
        for (int kw = kw_start; kw < kw_end; ++kw) {
            const int ix = in_x_base + kw * dilation;
            max_val = max(max_val, __ldg(&input[y_offset + ix]));
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

    dim3 threads(32, 8); // Optimal for coalesced width-first access
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA) with coalesced warps");
}