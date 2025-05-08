#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int const_params[8];

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
    __shared__ int shared_dims[4];
    if (threadIdx.x < 4) {
        shared_dims[threadIdx.x] = const_params[threadIdx.x];
    }
    __syncthreads();

    const int kernel_size = shared_dims[0];
    const int stride = shared_dims[1];
    const int padding = shared_dims[2];
    const int dilation = shared_dims[3];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * channels * output_height * output_width;
    const int elements_per_thread = (total_elements + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);

    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        const int output_idx = tid + i * blockDim.x * gridDim.x;
        if (output_idx >= total_elements) break;

        const int ow = output_idx % output_width;
        const int oh = (output_idx / output_width) % output_height;
        const int c = (output_idx / (output_width * output_height)) % channels;
        const int b = output_idx / (output_width * output_height * channels);

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        const int base_ih = oh * stride - padding;
        const int base_iw = ow * stride - padding;
        const int batch_offset = b * (channels * input_height * input_width);
        const int channel_offset = c * (input_height * input_width);

        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            const int ih = base_ih + kh * dilation;
            
            if (ih >= 0 && ih < input_height) {
                const int ih_offset = ih * input_width;
                
                #pragma unroll
                for (int kw = 0; kw < kernel_size; kw++) {
                    const int iw = base_iw + kw * dilation;
                    
                    if (iw >= 0 && iw < input_width) {
                        const int input_idx = batch_offset + channel_offset + ih_offset + iw;
                        max_val = max(max_val, __ldg(&input[input_idx]));
                    }
                }
            }
        }

        output[output_idx] = max_val;
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

    const int threads = 256;
    const int blocks = min(65535, (batch_size * channels * output_height * output_width + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
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