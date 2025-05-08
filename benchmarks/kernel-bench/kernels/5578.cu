#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int const_params[4];

template <typename scalar_t>
__device__ __forceinline__ int compute_input_index(
    int b, int c, int h, int w,
    int channels, int height, int width
) {
    return b * (channels * height * width) +
           c * (height * width) +
           h * width +
           w;
}

template <typename scalar_t>
__device__ __forceinline__ bool is_valid_input_location(
    int h, int w,
    int input_height, int input_width
) {
    return (h >= 0 && h < input_height && w >= 0 && w < input_width);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_max_pool(
    const scalar_t* __restrict__ input,
    int oh, int ow,
    int b, int c,
    int input_height, int input_width,
    int channels
) {
    const int kernel_size = const_params[0];
    const int stride = const_params[1];
    const int padding = const_params[2];
    const int dilation = const_params[3];

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        const int ih = oh * stride - padding + kh * dilation;
        
        #pragma unroll
        for (int kw = 0; kw < kernel_size; kw++) {
            const int iw = ow * stride - padding + kw * dilation;

            if (is_valid_input_location<scalar_t>(ih, iw, input_height, input_width)) {
                const int input_idx = compute_input_index<scalar_t>(
                    b, c, ih, iw,
                    channels, input_height, input_width
                );
                max_val = max(max_val, __ldg(&input[input_idx]));
            }
        }
    }

    return max_val;
}

template <typename scalar_t>
__global__ void optimized_max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_height * output_width) return;

    const int ow = idx % output_width;
    const int oh = (idx / output_width) % output_height;
    const int c = (idx / (output_width * output_height)) % channels;
    const int b = idx / (output_width * output_height * channels);

    const scalar_t max_val = compute_max_pool<scalar_t>(
        input, oh, ow, b, c,
        input_height, input_width, channels
    );

    output[idx] = max_val;
}

torch::Tensor optimized_max_pool2d_cuda_forward(
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

    const int params[4] = {kernel_size, stride, padding, dilation};
    cudaMemcpyToSymbol(const_params, params, sizeof(int) * 4);

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "optimized_max_pool2d_cuda_forward", ([&] {
        optimized_max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &optimized_max_pool2d_cuda_forward, "Optimized Max Pool 2D forward (CUDA)");
}