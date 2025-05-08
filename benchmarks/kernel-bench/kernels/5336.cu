#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

struct PoolParams {
    int batch_size;
    int channels;
    int input_height;
    int input_width;
    int output_height;
    int output_width;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
};

__constant__ PoolParams c_params;

// Optimized kernel with __ldg() and constant memory for parameters

template <typename scalar_t>
__global__ void max_pool2d_optimized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output
) {
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int total_output = c_params.batch_size * c_params.channels * 
                            c_params.output_height * c_params.output_width;
    if (output_idx >= total_output) return;

    const int ow = output_idx % c_params.output_width;
    const int oh = (output_idx / c_params.output_width) % c_params.output_height;
    const int c = (output_idx / (c_params.output_width * c_params.output_height)) % c_params.channels;
    const int b = output_idx / (c_params.output_width * c_params.output_height * c_params.channels);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    const int input_batch_offset = b * (c_params.channels * c_params.input_height * c_params.input_width);
    const int input_channel_offset = c * (c_params.input_height * c_params.input_width);

    #pragma unroll
    for (int kh = 0; kh < c_params.kernel_size; kh++) {
        const int ih = oh * c_params.stride - c_params.padding + kh * c_params.dilation;
        if (ih >= 0 && ih < c_params.input_height) {
            const int ih_offset = ih * c_params.input_width;

            #pragma unroll
            for (int kw = 0; kw < c_params.kernel_size; kw++) {
                const int iw = ow * c_params.stride - c_params.padding + kw * c_params.dilation;
                if (iw >= 0 && iw < c_params.input_width) {
                    const int input_idx = input_batch_offset + input_channel_offset + ih_offset + iw;
                    max_val = max(max_val, __ldg(&input[input_idx]));
                }
            }
        }
    }

    output[output_idx] = max_val;
}

torch::Tensor max_pool2d_cuda_forward_optimized(
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

    PoolParams host_params = {
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
    };

    cudaMemcpyToSymbol(c_params, &host_params, sizeof(PoolParams));

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward_optimized", ([&] {
        max_pool2d_optimized_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward_optimized, "Max Pool 2D forward optimized (CUDA)");
}