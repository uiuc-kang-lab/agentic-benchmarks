#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_kernel_strided(
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
    const int dilation,
    const int elements_per_thread
) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int total_elements = batch_size * channels * output_height * output_width;

    // Each thread processes multiple elements strided by total number of threads
    for (int idx = thread_id; idx < total_elements; idx += total_threads) {
        const int ow = idx % output_width;
        const int oh = (idx / output_width) % output_height;
        const int c = (idx / (output_width * output_height)) % channels;
        const int b = idx / (output_width * output_height * channels);

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        if (kernel_size == 2) {
            // Manually unrolled for 2x2 kernel
            #pragma unroll
            for (int kh = 0; kh < 2; kh++) {
                #pragma unroll
                for (int kw = 0; kw < 2; kw++) {
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
        } else {
            for (int kh = 0; kh < kernel_size; kh++) {
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
        }

        output[idx] = max_val;
    }
}

torch::Tensor max_pool2d_cuda_forward_strided(
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

    // Calculate optimal thread configuration
    const int threads_per_block = 256;
    const int elements_per_thread = 4; // Each thread processes multiple elements
    const int total_elements = batch_size * channels * output_height * output_width;
    const int num_blocks = (total_elements + threads_per_block * elements_per_thread - 1) / 
                          (threads_per_block * elements_per_thread);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward_strided", ([&] {
        max_pool2d_kernel_strided<scalar_t><<<num_blocks, threads_per_block>>>(
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
            dilation,
            elements_per_thread
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward_strided, "Max Pool 2D forward with strided processing (CUDA)");
}