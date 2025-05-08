#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Constant memory for frequently accessed parameters
__constant__ int c_input_dims[4];  // batch_size, channels, input_height, input_width
__constant__ int c_output_dims[2]; // output_height, output_width
__constant__ int c_pool_params[4]; // kernel_size, stride, padding, dilation

template <typename scalar_t>
__global__ void constant_mem_maxpool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output
) {
    const int total_elements = c_input_dims[0] * c_input_dims[1] * c_output_dims[0] * c_output_dims[1];
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += blockDim.x * gridDim.x) {
        
        // Calculate position using constant memory dimensions
        const int ow = idx % c_output_dims[1];
        const int oh = (idx / c_output_dims[1]) % c_output_dims[0];
        const int c  = (idx / (c_output_dims[1] * c_output_dims[0])) % c_input_dims[1];
        const int b  = idx / (c_output_dims[1] * c_output_dims[0] * c_input_dims[1]);

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        // Calculate input plane offset once
        const int input_plane_size = c_input_dims[2] * c_input_dims[3];
        const int batch_offset = b * c_input_dims[1] * input_plane_size;
        const int channel_offset = c * input_plane_size;

        // Pooling window iteration using constant memory parameters
        #pragma unroll
        for (int kh = 0; kh < c_pool_params[0]; kh++) {
            #pragma unroll
            for (int kw = 0; kw < c_pool_params[0]; kw++) {
                const int ih = oh * c_pool_params[1] - c_pool_params[2] + kh * c_pool_params[3];
                const int iw = ow * c_pool_params[1] - c_pool_params[2] + kw * c_pool_params[3];

                if (ih >= 0 && ih < c_input_dims[2] && iw >= 0 && iw < c_input_dims[3]) {
                    const int input_idx = batch_offset + channel_offset + ih * c_input_dims[3] + iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
        output[idx] = max_val;
    }
}

torch::Tensor constant_mem_maxpool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    // Copy parameters to constant memory
    int input_dims[4] = {batch_size, channels, input_height, input_width};
    int output_dims[2] = {output_height, output_width};
    int pool_params[4] = {kernel_size, stride, padding, dilation};
    
    cudaMemcpyToSymbol(c_input_dims, input_dims, sizeof(int) * 4);
    cudaMemcpyToSymbol(c_output_dims, output_dims, sizeof(int) * 2);
    cudaMemcpyToSymbol(c_pool_params, pool_params, sizeof(int) * 4);

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "constant_mem_maxpool2d_cuda_forward", ([&] {
        constant_mem_maxpool2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>()
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &constant_mem_maxpool2d_cuda_forward, "Constant Memory Max Pool 2D forward (CUDA)");
}