#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int const_params[8];

template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    // Use 2D block configuration for better occupancy
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;
    const int bid = blockIdx.x;
    
    const int total_elements = batch_size * channels * output_height * output_width;
    
    // Each thread processes multiple elements strided by total threads
    for (int idx = bid * block_size + tid; idx < total_elements; idx += gridDim.x * block_size) {
        const int kernel_size = const_params[0];
        const int stride = const_params[1];
        const int padding = const_params[2];
        const int dilation = const_params[3];

        const int ow = idx % output_width;
        const int oh = (idx / output_width) % output_height;
        const int c = (idx / (output_width * output_height)) % channels;
        const int b = idx / (output_width * output_height * channels);

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            const int ih = oh * stride - padding + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                #pragma unroll
                for (int kw = 0; kw < kernel_size; kw++) {
                    const int iw = ow * stride - padding + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        const int input_idx = b * (channels * input_height * input_width) +
                                            c * (input_height * input_width) +
                                            ih * input_width +
                                            iw;
                        max_val = max(max_val, __ldg(&input[input_idx]));
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

    const int params[8] = {kernel_size, stride, padding, dilation};
    cudaMemcpyToSymbol(const_params, params, sizeof(int) * 8);

    // Use 16x16 thread block for better occupancy
    const dim3 threads(16, 16);
    const int total_elements = batch_size * channels * output_height * output_width;
    const int blocks = (total_elements + (threads.x * threads.y) - 1) / (threads.x * threads.y);
    
    // Limit number of blocks to avoid overhead
    const int max_blocks = 65535;
    const int num_blocks = min(blocks, max_blocks);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<num_blocks, threads>>>(
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