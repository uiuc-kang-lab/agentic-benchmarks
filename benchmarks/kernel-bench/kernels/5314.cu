#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Declare constant memory for frequently accessed parameters
__constant__ int c_input_dims[4];    // batch_size, channels, height, width
__constant__ int c_output_dims[4];   // batch_size, channels, height, width
__constant__ int c_pool_params[4];   // kernel_size, stride, padding, dilation

template <typename scalar_t>
__global__ void max_pool2d_kernel_constant(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output
) {
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = c_output_dims[0] * c_output_dims[1] * c_output_dims[2] * c_output_dims[3];
    
    if (output_idx >= total_outputs) return;

    const int ow = output_idx % c_output_dims[3];
    const int oh = (output_idx / c_output_dims[3]) % c_output_dims[2];
    const int c = (output_idx / (c_output_dims[3] * c_output_dims[2])) % c_output_dims[1];
    const int b = output_idx / (c_output_dims[3] * c_output_dims[2] * c_output_dims[1]);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    #pragma unroll
    for (int kh = 0; kh < c_pool_params[0]; kh++) {
        const int ih = oh * c_pool_params[1] - c_pool_params[2] + kh * c_pool_params[3];
        
        if (ih >= 0 && ih < c_input_dims[2]) {
            #pragma unroll
            for (int kw = 0; kw < c_pool_params[0]; kw++) {
                const int iw = ow * c_pool_params[1] - c_pool_params[2] + kw * c_pool_params[3];
                
                if (iw >= 0 && iw < c_input_dims[3]) {
                    const int input_idx = b * (c_input_dims[1] * c_input_dims[2] * c_input_dims[3]) +
                                        c * (c_input_dims[2] * c_input_dims[3]) +
                                        ih * c_input_dims[3] +
                                        iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    }

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

    // Copy dimensions to constant memory
    int input_dims[4] = {batch_size, channels, input_height, input_width};
    int output_dims[4] = {batch_size, channels, output_height, output_width};
    int pool_params[4] = {kernel_size, stride, padding, dilation};
    
    cudaMemcpyToSymbol(c_input_dims, input_dims, sizeof(int) * 4);
    cudaMemcpyToSymbol(c_output_dims, output_dims, sizeof(int) * 4);
    cudaMemcpyToSymbol(c_pool_params, pool_params, sizeof(int) * 4);

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel_constant<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}