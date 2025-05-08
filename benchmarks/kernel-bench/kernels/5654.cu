#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Constant memory for frequently accessed parameters
__constant__ int KERNEL_PARAMS[8];  // batch_size, channels, input_height, input_width, output_height, output_width, kernel_size, stride

template <typename scalar_t>
__global__ void coalesced_maxpool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int padding,
    const int dilation) {
    
    // Get kernel parameters from constant memory
    const int batch_size = KERNEL_PARAMS[0];
    const int channels = KERNEL_PARAMS[1];
    const int input_height = KERNEL_PARAMS[2];
    const int input_width = KERNEL_PARAMS[3];
    const int output_height = KERNEL_PARAMS[4];
    const int output_width = KERNEL_PARAMS[5];
    const int kernel_size = KERNEL_PARAMS[6];
    const int stride = KERNEL_PARAMS[7];

    // Map blocks to batch and channel dimensions
    const int b = blockIdx.z % batch_size;
    const int c = blockIdx.z / batch_size;
    
    // Map remaining threads to height and width
    const int oh = blockIdx.y;
    const int ow_base = threadIdx.x;
    
    // Skip if this block is out of bounds
    if (c >= channels || oh >= output_height) return;

    // Input offset for current batch and channel
    const int input_batch_offset = b * channels * input_height * input_width;
    const int input_channel_offset = c * input_height * input_width;
    const int input_base = input_batch_offset + input_channel_offset;

    // Output offset for current batch and channel
    const int output_batch_offset = b * channels * output_height * output_width;
    const int output_channel_offset = c * output_height * output_width;
    const int output_row_offset = oh * output_width;
    
    // Process multiple elements along width dimension with grid stride
    for (int ow = ow_base; ow < output_width; ow += blockDim.x) {
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        // Calculate input window bounds
        const int ih_start = oh * stride - padding;
        const int iw_start = ow * stride - padding;
        
        // Process pooling window with coalesced access pattern
        #pragma unroll
        for (int kh = 0; kh < kernel_size; ++kh) {
            const int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                const int input_row_offset = ih * input_width;
                
                #pragma unroll
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int iw = iw_start + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        const int input_idx = input_base + input_row_offset + iw;
                        max_val = max(max_val, input[input_idx]);
                    }
                }
            }
        }
        
        // Write output with coalesced access
        const int output_idx = output_batch_offset + output_channel_offset + output_row_offset + ow;
        output[output_idx] = max_val;
    }
}

torch::Tensor coalesced_maxpool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {
    
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    
    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    
    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());
    
    // Copy parameters to constant memory
    int h_kernel_params[8] = {batch_size, channels, input_height, input_width,
                             output_height, output_width, kernel_size, stride};
    cudaMemcpyToSymbol(KERNEL_PARAMS, h_kernel_params, sizeof(int) * 8);
    
    // Configure kernel launch parameters for coalesced access
    const int threads_x = 256;  // Process consecutive elements along width
    const dim3 threads(threads_x);
    const dim3 blocks(1, output_height, batch_size * channels);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "coalesced_maxpool2d_cuda_forward", ([&] {
        coalesced_maxpool2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            padding,
            dilation
        );
    }));
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &coalesced_maxpool2d_cuda_forward, "Coalesced Max Pool 2D forward (CUDA)");
}