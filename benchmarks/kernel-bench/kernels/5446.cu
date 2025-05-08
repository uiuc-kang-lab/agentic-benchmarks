#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_kernel_warp(
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
    const unsigned int warp_size = 32;
    const unsigned int lane_id = threadIdx.x % warp_size;
    const unsigned int warp_id = threadIdx.x / warp_size;
    const unsigned int warps_per_block = blockDim.x / warp_size;
    const unsigned int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    
    const unsigned int total_output_elements = batch_size * channels * output_height * output_width;
    const unsigned int elements_per_warp = (total_output_elements + gridDim.x * warps_per_block - 1) / 
                                         (gridDim.x * warps_per_block);
    
    const unsigned int warp_start = global_warp_id * elements_per_warp;
    const unsigned int warp_end = min(warp_start + elements_per_warp, total_output_elements);

    for (unsigned int output_idx = warp_start + lane_id; 
         output_idx < warp_end; 
         output_idx += warp_size) {
        
        if (output_idx >= total_output_elements) continue;

        const int ow = output_idx % output_width;
        const int oh = (output_idx / output_width) % output_height;
        const int c = (output_idx / (output_width * output_height)) % channels;
        const int b = output_idx / (output_width * output_height * channels);

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
                    max_val = max(max_val, __ldg(&input[input_idx]));
                }
            }
        }

        if (output_idx < total_output_elements) {
            output[output_idx] = max_val;
        }
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

    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int num_blocks = (batch_size * channels * output_height * output_width + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel_warp<scalar_t><<<num_blocks, threads_per_block>>>(
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