#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Constant memory for frequently accessed parameters
__constant__ int c_input_dims[4];  // batch_size, channels, input_height, input_width
__constant__ int c_output_dims[4]; // batch_size, channels, output_height, output_width
__constant__ int c_pool_params[4]; // kernel_size, stride, padding, dilation
__constant__ int c_strides[4];     // input_batch_stride, input_channel_stride, output_batch_stride, output_channel_stride

template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = c_output_dims[0] * c_output_dims[1] * c_output_dims[2] * c_output_dims[3];
    
    if (tid >= total_elements) return;

    const int ow = tid % c_output_dims[3];
    const int oh = (tid / c_output_dims[3]) % c_output_dims[2];
    const int c = (tid / (c_output_dims[3] * c_output_dims[2])) % c_output_dims[1];
    const int b = tid / (c_output_dims[3] * c_output_dims[2] * c_output_dims[1]);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    const int base_input_offset = b * c_strides[0] + c * c_strides[1];

    #pragma unroll
    for (int kh = 0; kh < c_pool_params[0]; kh++) {
        const int ih = oh * c_pool_params[1] - c_pool_params[2] + kh * c_pool_params[3];
        
        if (ih >= 0 && ih < c_input_dims[2]) {
            const int ih_offset = ih * c_input_dims[3];
            
            #pragma unroll
            for (int kw = 0; kw < c_pool_params[0]; kw++) {
                const int iw = ow * c_pool_params[1] - c_pool_params[2] + kw * c_pool_params[3];
                
                if (iw >= 0 && iw < c_input_dims[3]) {
                    const int input_idx = base_input_offset + ih_offset + iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    }

    const int output_idx = b * c_strides[2] + c * c_strides[3] + oh * c_output_dims[3] + ow;
    output[output_idx] = max_val;
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

    // Prepare constant memory data
    int h_input_dims[4] = {batch_size, channels, input_height, input_width};
    int h_output_dims[4] = {batch_size, channels, output_height, output_width};
    int h_pool_params[4] = {kernel_size, stride, padding, dilation};
    
    int h_strides[4] = {
        channels * input_height * input_width,    // input_batch_stride
        input_height * input_width,               // input_channel_stride
        channels * output_height * output_width,  // output_batch_stride
        output_height * output_width             // output_channel_stride
    };

    // Copy to constant memory
    cudaMemcpyToSymbol(c_input_dims, h_input_dims, sizeof(int) * 4);
    cudaMemcpyToSymbol(c_output_dims, h_output_dims, sizeof(int) * 4);
    cudaMemcpyToSymbol(c_pool_params, h_pool_params, sizeof(int) * 4);
    cudaMemcpyToSymbol(c_strides, h_strides, sizeof(int) * 4);

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int total_elements = batch_size * channels * output_height * output_width;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}