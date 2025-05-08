#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void max_pool2d_warp_uniform_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    const int warp_idx = blockIdx.x * warps_per_block + warp_id;
    
    const int total_warps = gridDim.x * warps_per_block;
    const int total_elements = batch_channels * output_height * output_width;
    
    // Grid-stride loop over warps
    for (int base_idx = warp_idx * warp_size; base_idx < total_elements; base_idx += total_warps * warp_size) {
        const int idx = base_idx + lane_id;
        
        if (idx < total_elements) {
            const int bc = idx / (output_height * output_width);
            const int oh = (idx / output_width) % output_height;
            const int ow = idx % output_width;
            
            // Calculate input boundaries for this output position
            const int ih_start = oh * stride - padding;
            const int iw_start = ow * stride - padding;
            const int ih_end = min(ih_start + kernel_size * dilation, input_height);
            const int iw_end = min(iw_start + kernel_size * dilation, input_width);
            
            scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
            
            // Uniform control flow for kernel size 2
            if (kernel_size == 2) {
                #pragma unroll
                for (int kh = 0; kh < 2; ++kh) {
                    const int ih = ih_start + kh * dilation;
                    if (ih >= 0 && ih < input_height) {
                        #pragma unroll
                        for (int kw = 0; kw < 2; ++kw) {
                            const int iw = iw_start + kw * dilation;
                            if (iw >= 0 && iw < input_width) {
                                const int input_idx = (bc * input_height + ih) * input_width + iw;
                                max_val = max(max_val, input[input_idx]);
                            }
                        }
                    }
                }
            } else {
                // Uniform control flow for other kernel sizes
                for (int ih = max(0, ih_start); ih < ih_end; ih += dilation) {
                    for (int iw = max(0, iw_start); iw < iw_end; iw += dilation) {
                        const int input_idx = (bc * input_height + ih) * input_width + iw;
                        max_val = max(max_val, input[input_idx]);
                    }
                }
            }
            
            output[idx] = max_val;
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
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    
    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());
    
    const int batch_channels = batch_size * channels;
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int num_blocks = (batch_channels * output_height * output_width + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_forward", ([&] {
        max_pool2d_warp_uniform_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_channels,
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with warp uniform execution (CUDA)");
}