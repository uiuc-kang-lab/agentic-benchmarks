#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int constant_params[4];  // [kernel_size, stride, padding, dilation]

template <typename scalar_t>
__global__ void max_pool2d_constant_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    const int kernel_size = constant_params[0];
    const int stride = constant_params[1];
    const int padding = constant_params[2];
    const int dilation = constant_params[3];

    // Use 2D block for better spatial locality
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Each block handles a 32x8 tile of output
    const int ow = bx * 32 + tx;
    const int oh = by * 8 + ty;
    
    // Handle multiple channels per thread block
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            if (oh < output_height && ow < output_width) {
                scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
                
                #pragma unroll
                for (int kh = 0; kh < kernel_size; kh++) {
                    #pragma unroll
                    for (int kw = 0; kw < kernel_size; kw++) {
                        const int ih = oh * stride - padding + kh * dilation;
                        const int iw = ow * stride - padding + kw * dilation;

                        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                            const int input_idx = b * (channels * input_height * input_width) +
                                                c * (input_height * input_width) +
                                                ih * input_width +
                                                iw;
                            max_val = max(max_val, input[input_idx]);
                        }
                    }
                }

                const int output_idx = b * (channels * output_height * output_width) +
                                     c * (output_height * output_width) +
                                     oh * output_width +
                                     ow;
                output[output_idx] = max_val;
            }
        }
    }
}

torch::Tensor max_pool2d_constant_cuda_forward(
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

    int params[4] = {kernel_size, stride, padding, dilation};
    cudaMemcpyToSymbol(constant_params, params, sizeof(params));

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_constant_cuda_forward", ([&] {
        max_pool2d_constant_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_constant_cuda_forward, "Max Pool 2D forward (CUDA)");
}