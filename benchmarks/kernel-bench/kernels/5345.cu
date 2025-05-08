#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for frequently accessed parameters
__constant__ int c_dims[8];  // batch_size, channels, input_h, input_w, output_h, output_w, stride, padding
__constant__ int c_kernel_params[2];  // kernel_size, dilation

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_constant_mem_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output
) {
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= c_dims[0] * c_dims[1] * c_dims[4] * c_dims[5]) return;

    const int ow = output_idx % c_dims[5];
    const int oh = (output_idx / c_dims[5]) % c_dims[4];
    const int c = (output_idx / (c_dims[5] * c_dims[4])) % c_dims[1];
    const int b = output_idx / (c_dims[5] * c_dims[4] * c_dims[1]);

    const int input_batch_offset = b * (c_dims[1] * c_dims[2] * c_dims[3]);
    const int input_channel_offset = c * (c_dims[2] * c_dims[3]);
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    if constexpr (KERNEL_SIZE == 2) {
        const int ih_base = oh * c_dims[6] - c_dims[7];
        const int iw_base = ow * c_dims[6] - c_dims[7];
        
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int ih = ih_base + i * c_kernel_params[1];
            if (ih >= 0 && ih < c_dims[2]) {
                const int ih_offset = ih * c_dims[3];
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int iw = iw_base + j * c_kernel_params[1];
                    if (iw >= 0 && iw < c_dims[3]) {
                        const int idx = input_batch_offset + input_channel_offset + ih_offset + iw;
                        max_val = max(max_val, __ldg(&input[idx]));
                    }
                }
            }
        }
    }
    else if constexpr (KERNEL_SIZE == 3) {
        const int ih_base = oh * c_dims[6] - c_dims[7];
        const int iw_base = ow * c_dims[6] - c_dims[7];
        
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            const int ih = ih_base + i * c_kernel_params[1];
            if (ih >= 0 && ih < c_dims[2]) {
                const int ih_offset = ih * c_dims[3];
                #pragma unroll
                for (int j = 0; j < 3; j++) {
                    const int iw = iw_base + j * c_kernel_params[1];
                    if (iw >= 0 && iw < c_dims[3]) {
                        const int idx = input_batch_offset + input_channel_offset + ih_offset + iw;
                        max_val = max(max_val, __ldg(&input[idx]));
                    }
                }
            }
        }
    }
    else {
        const int ih_base = oh * c_dims[6] - c_dims[7];
        const int iw_base = ow * c_dims[6] - c_dims[7];
        
        for (int i = 0; i < c_kernel_params[0]; i++) {
            const int ih = ih_base + i * c_kernel_params[1];
            if (ih >= 0 && ih < c_dims[2]) {
                const int ih_offset = ih * c_dims[3];
                for (int j = 0; j < c_kernel_params[0]; j++) {
                    const int iw = iw_base + j * c_kernel_params[1];
                    if (iw >= 0 && iw < c_dims[3]) {
                        const int idx = input_batch_offset + input_channel_offset + ih_offset + iw;
                        max_val = max(max_val, __ldg(&input[idx]));
                    }
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

    // Copy parameters to constant memory
    int h_dims[8] = {batch_size, channels, input_height, input_width, 
                     output_height, output_width, stride, padding};
    int h_kernel_params[2] = {kernel_size, dilation};
    
    cudaMemcpyToSymbol(c_dims, h_dims, sizeof(int) * 8);
    cudaMemcpyToSymbol(c_kernel_params, h_kernel_params, sizeof(int) * 2);

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int threads = 128;
    const int total_elements = batch_size * channels * output_height * output_width;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            max_pool2d_constant_mem_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>());
        }
        else if (kernel_size == 3) {
            max_pool2d_constant_mem_kernel<scalar_t, 3><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>());
        }
        else {
            max_pool2d_constant_mem_kernel<scalar_t, -1><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>());
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with constant memory (CUDA)");
}