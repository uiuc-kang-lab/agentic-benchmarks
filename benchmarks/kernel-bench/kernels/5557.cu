#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int const_params[8];

template <typename scalar_t>
__global__ void max_pool2d_coalesced_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    const int batch_channel_idx = blockIdx.z * channels + blockIdx.y;
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;

    if (ow >= output_width || batch_channel_idx >= batch_size * channels) return;

    const int kernel_size = const_params[0];
    const int stride = const_params[1];
    const int padding = const_params[2];
    const int dilation = const_params[3];

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    for (int kh = 0; kh < kernel_size; kh++) {
        const int oh = blockIdx.y * stride - padding + kh * dilation;
        if (oh < 0 || oh >= input_height) continue;

        for (int kw = 0; kw < kernel_size; kw++) {
            const int iw = ow * stride - padding + kw * dilation;
            if (iw < 0 || iw >= input_width) continue;

            int input_idx = batch_channel_idx * input_height * input_width + oh * input_width + iw;
            max_val = max(max_val, __ldg(&input[input_idx]));
        }
    }

    int output_idx = batch_channel_idx * output_height * output_width + blockIdx.y * output_width + ow;
    output[output_idx] = max_val;
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

    const int params[8] = {kernel_size, stride, padding, dilation};
    cudaMemcpyToSymbol(const_params, params, sizeof(int) * 8);

    const int threads = 128;
    const dim3 blocks((output_width + threads - 1) / threads, output_height, batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_coalesced_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}
