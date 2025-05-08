#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__device__ __forceinline__ void get_pool_indices(
    const int output_idx,
    const int output_width,
    const int output_height,
    const int channels,
    int& ow,
    int& oh,
    int& c,
    int& b
) {
    ow = output_idx % output_width;
    oh = (output_idx / output_width) % output_height;
    c = (output_idx / (output_width * output_height)) % channels;
    b = output_idx / (output_width * output_height * channels);
}

template <typename scalar_t>
__device__ __forceinline__ bool is_valid_input_index(
    const int ih,
    const int iw,
    const int input_height,
    const int input_width
) {
    return (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_max_value(
    const scalar_t* input,
    const int b,
    const int c,
    const int oh,
    const int ow,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int input_height,
    const int input_width,
    const int channel_stride
) {
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const int base_input_offset = b * channel_stride + c * (input_height * input_width);

    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        const int ih = oh * stride - padding + kh * dilation;
        
        #pragma unroll
        for (int kw = 0; kw < kernel_size; kw++) {
            const int iw = ow * stride - padding + kw * dilation;
            
            if (is_valid_input_index<scalar_t>(ih, iw, input_height, input_width)) {
                const int input_idx = base_input_offset + ih * input_width + iw;
                max_val = max(max_val, input[input_idx]);
            }
        }
    }
    return max_val;
}

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
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    int ow, oh, c, b;
    get_pool_indices<scalar_t>(output_idx, output_width, output_height, channels, ow, oh, c, b);

    const int channel_stride = channels * input_height * input_width;
    output[output_idx] = compute_max_value<scalar_t>(
        input, b, c, oh, ow,
        kernel_size, stride, padding, dilation,
        input_height, input_width, channel_stride
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

    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}