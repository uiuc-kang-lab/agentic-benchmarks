#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int pool_params[6];  // batch_size, channels, input_height, input_width, stride, padding

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_warp_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int output_height,
    const int output_width,
    const int dilation
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int global_warp_id = (blockIdx.x * warps_per_block) + warp_id;
    
    const int total_elements = pool_params[0] * pool_params[1] * output_height * output_width;
    const int total_warps = (total_elements + 31) >> 5;
    
    for (int warp_offset = global_warp_id; warp_offset < total_warps; warp_offset += gridDim.x * warps_per_block) {
        const int thread_idx = (warp_offset << 5) + lane_id;
        
        if (thread_idx < total_elements) {
            const int ow = thread_idx & (output_width - 1);
            const int oh = (thread_idx / output_width) & (output_height - 1);
            const int c = (thread_idx / (output_width * output_height)) & (pool_params[1] - 1);
            const int b = thread_idx / (output_width * output_height * pool_params[1]);
            
            const int input_idx_base = (b * pool_params[1] + c) * pool_params[2] * pool_params[3];
            const int ih_base = oh * pool_params[4] - pool_params[5];
            const int iw_base = ow * pool_params[4] - pool_params[5];
            
            scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
            
            const int ih_valid_start = max(0, ih_base);
            const int ih_valid_end = min(pool_params[2], ih_base + KERNEL_SIZE * dilation);
            const int iw_valid_start = max(0, iw_base);
            const int iw_valid_end = min(pool_params[3], iw_base + KERNEL_SIZE * dilation);
            
            #pragma unroll
            for (int ih = ih_valid_start; ih < ih_valid_end; ih += dilation) {
                const int row_offset = input_idx_base + ih * pool_params[3];
                #pragma unroll
                for (int iw = iw_valid_start; iw < iw_valid_end; iw += dilation) {
                    max_val = max(max_val, __ldg(&input[row_offset + iw]));
                }
            }
            
            output[thread_idx] = max_val;
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
    
    int h_pool_params[6] = {batch_size, channels, input_height, input_width, stride, padding};
    cudaMemcpyToSymbol(pool_params, h_pool_params, sizeof(int) * 6);
    
    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());
    
    const int threads_per_block = 256;
    const int total_elements = batch_size * channels * output_height * output_width;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            max_pool2d_warp_kernel<scalar_t, 2><<<num_blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                output_height, output_width,
                dilation
            );
        }
        else if (kernel_size == 3) {
            max_pool2d_warp_kernel<scalar_t, 3><<<num_blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                output_height, output_width,
                dilation
            );
        }
        else {
            max_pool2d_warp_kernel<scalar_t, -1><<<num_blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                output_height, output_width,
                dilation
            );
        }
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}