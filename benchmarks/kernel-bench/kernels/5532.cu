#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_single_output(
    const scalar_t* __restrict__ input,
    const int b,
    const int c,
    const int oh,
    const int ow,
    const int input_height,
    const int input_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int input_stride_batch,
    const int input_stride_channel
) {
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const int base_offset = b * input_stride_batch + c * input_stride_channel;

    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        const int ih = oh * stride - padding + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            const int h_offset = ih * input_width;
            #pragma unroll
            for (int kw = 0; kw < kernel_size; kw++) {
                const int iw = ow * stride - padding + kw * dilation;
                if (iw >= 0 && iw < input_width) {
                    max_val = max(max_val, __ldg(&input[base_offset + h_offset + iw]));
                }
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
    const int input_stride_batch = channels * input_height * input_width;
    const int input_stride_channel = input_height * input_width;
    const int output_stride_batch = channels * output_height * output_width;
    const int output_stride_channel = output_height * output_width;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int total_elements = batch_size * channels * output_height * output_width;

    // Each thread processes multiple elements in a strided fashion
    for (int idx = tid; idx < total_elements; idx += total_threads) {
        const int ow = idx % output_width;
        const int oh = (idx / output_width) % output_height;
        const int c = (idx / (output_width * output_height)) % channels;
        const int b = idx / (output_width * output_height * channels);

        output[idx] = compute_single_output<scalar_t>(
            input, b, c, oh, ow,
            input_height, input_width,
            kernel_size, stride, padding, dilation,
            input_stride_batch, input_stride_channel
        );
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

    // Adjust thread and block configuration for strided access
    const int threads = 256;
    const int max_blocks = 65535;
    const int total_elements = batch_size * channels * output_height * output_width;
    const int elements_per_thread = (total_elements + threads * max_blocks - 1) / (threads * max_blocks);
    const int blocks = min((total_elements + threads * elements_per_thread - 1) / (threads * elements_per_thread), max_blocks);

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