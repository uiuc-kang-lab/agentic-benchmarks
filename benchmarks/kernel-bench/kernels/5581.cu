#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int const_params[4];

template <typename scalar_t>
__global__ void max_pool2d_aligned_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = batch_size * channels * output_height * output_width;
    
    // Process multiple elements per thread if needed
    for (int idx = tid; idx < total_threads; idx += blockDim.x * gridDim.x) {
        const int ow = idx % output_width;
        const int oh = (idx / output_width) % output_height;
        const int c = (idx / (output_width * output_height)) % channels;
        const int b = idx / (output_width * output_height * channels);

        const int kernel_size = const_params[0];
        const int stride = const_params[1];
        const int padding = const_params[2];
        const int dilation = const_params[3];

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        // Calculate base input offset for the current batch and channel
        const int batch_offset = b * channels * input_height * input_width;
        const int channel_offset = c * input_height * input_width;
        const scalar_t* input_ptr = input + batch_offset + channel_offset;

        // Aligned loads for kernel window
        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            const int ih = oh * stride - padding + kh * dilation;
            
            if (ih >= 0 && ih < input_height) {
                const int row_offset = ih * input_width;
                
                #pragma unroll
                for (int kw = 0; kw < kernel_size; kw++) {
                    const int iw = ow * stride - padding + kw * dilation;
                    
                    if (iw >= 0 && iw < input_width) {
                        max_val = max(max_val, __ldg(input_ptr + row_offset + iw));
                    }
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

    const int params[4] = {kernel_size, stride, padding, dilation};
    cudaMemcpyToSymbol(const_params, params, sizeof(int) * 4);

    const int threads = 128;
    const int max_blocks = 65535;
    const int total_elements = batch_size * channels * output_height * output_width;
    const int blocks = min(max_blocks, (total_elements + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_aligned_kernel<scalar_t><<<blocks, threads>>>(
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