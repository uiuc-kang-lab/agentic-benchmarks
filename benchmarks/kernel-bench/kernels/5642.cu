#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void optimized_maxpool2d_kernel(
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
    extern __shared__ scalar_t shared_pool[];
    
    const int total_elements = batch_size * channels * output_height * output_width;
    const int tid = threadIdx.x;
    
    for (int out_idx = blockIdx.x * blockDim.x + tid; 
         out_idx < total_elements; 
         out_idx += blockDim.x * gridDim.x) {
        
        const int ow = out_idx % output_width;
        const int oh = (out_idx / output_width) % output_height;
        const int c  = (out_idx / (output_width * output_height)) % channels;
        const int b  = out_idx / (output_width * output_height * channels);

        scalar_t max_val = -__FLT_MAX__;

        const int ih_start = oh * stride - padding;
        const int iw_start = ow * stride - padding;
        
        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            const int ih = ih_start + kh * dilation;
            
            if (ih >= 0 && ih < input_height) {
                #pragma unroll
                for (int kw = 0; kw < kernel_size; kw++) {
                    const int iw = iw_start + kw * dilation;
                    
                    if (iw >= 0 && iw < input_width) {
                        const int input_idx = b * (channels * input_height * input_width) +
                                            c * (input_height * input_width) +
                                            ih * input_width +
                                            iw;
                        max_val = max(max_val, input[input_idx]);
                    }
                }
            }
        }
        
        output[out_idx] = max_val;
    }
}

torch::Tensor optimized_maxpool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int max_blocks = 65535;
    const int num_elements = batch_size * channels * output_height * output_width;
    const int blocks = min(max_blocks, (num_elements + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_maxpool2d_cuda_forward", ([&] {
        optimized_maxpool2d_kernel<scalar_t><<<blocks, threads>>>(
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

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_maxpool2d_cuda_forward, "Optimized Max Pool 2D forward (CUDA)");
}