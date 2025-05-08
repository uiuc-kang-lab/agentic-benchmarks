#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int d_kernel_params[12];

template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output
) {
    extern __shared__ scalar_t shared_input[];
    
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = d_kernel_params[0] * d_kernel_params[1] * 
                             d_kernel_params[4] * d_kernel_params[5];
    
    if (output_idx >= total_elements) return;

    const int ow = output_idx % d_kernel_params[5];
    const int oh = (output_idx / d_kernel_params[5]) % d_kernel_params[4];
    const int c = (output_idx / (d_kernel_params[5] * d_kernel_params[4])) % d_kernel_params[1];
    const int b = output_idx / (d_kernel_params[5] * d_kernel_params[4] * d_kernel_params[1]);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    const int ih_start = oh * d_kernel_params[7] - d_kernel_params[8];
    const int iw_start = ow * d_kernel_params[7] - d_kernel_params[8];
    const int ih_end = ih_start + d_kernel_params[6] * d_kernel_params[9];
    const int iw_end = iw_start + d_kernel_params[6] * d_kernel_params[9];

    const int input_stride_batch = d_kernel_params[10];
    const int base_input_offset = b * input_stride_batch + 
                                 c * d_kernel_params[2] * d_kernel_params[3];

    using vec4_t = typename cuda::aligned_vector<scalar_t, 4>::type;
    
    #pragma unroll 4
    for (int ih = ih_start; ih < ih_end; ih += d_kernel_params[9]) {
        if (ih >= 0 && ih < d_kernel_params[2]) {
            const int ih_offset = ih * d_kernel_params[3];
            
            #pragma unroll 4
            for (int iw = iw_start; iw < iw_end; iw += d_kernel_params[9]) {
                if (iw >= 0 && iw < d_kernel_params[3]) {
                    const int input_idx = base_input_offset + ih_offset + iw;
                    max_val = max(max_val, __ldg(&input[input_idx]));
                }
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    if (threadIdx.x % 32 == 0) {
        const int output_write_idx = b * d_kernel_params[11] + 
                                   c * d_kernel_params[4] * d_kernel_params[5] + 
                                   oh * d_kernel_params[5] + 
                                   ow;
        output[output_write_idx] = max_val;
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

    const int input_stride = channels * input_height * input_width;
    const int output_stride = channels * output_height * output_width;

    int h_kernel_params[12] = {
        (int)batch_size, (int)channels, (int)input_height, (int)input_width,
        (int)output_height, (int)output_width, kernel_size, stride,
        padding, dilation, input_stride, output_stride
    };
    cudaMemcpyToSymbol(d_kernel_params, h_kernel_params, sizeof(int) * 12);

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;
    
    const int shared_mem_size = (kernel_size * kernel_size + threads) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>()
        );
    }));

    return output;
}