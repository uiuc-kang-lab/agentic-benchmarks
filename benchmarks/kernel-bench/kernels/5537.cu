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

    const int start_ih = oh * d_kernel_params[7] - d_kernel_params[8];
    const int start_iw = ow * d_kernel_params[7] - d_kernel_params[8];
    
    const int input_stride_batch = d_kernel_params[10];
    const int base_input_offset = b * input_stride_batch + 
                                 c * d_kernel_params[2] * d_kernel_params[3];

    using vec4_t = typename cuda::aligned_vector<scalar_t, 4>::type;
    
    #pragma unroll
    for (int kh = 0; kh < d_kernel_params[6]; kh++) {
        const int ih = start_ih + kh * d_kernel_params[9];
        if (ih >= 0 && ih < d_kernel_params[2]) {
            const int ih_offset = ih * d_kernel_params[3];
            
            #pragma unroll
            for (int kw = 0; kw < d_kernel_params[6]; kw += 4) {
                const int iw = start_iw + kw * d_kernel_params[9];
                
                if (iw >= 0 && iw + 3 < d_kernel_params[3] && d_kernel_params[9] == 1) {
                    const int input_idx = base_input_offset + ih_offset + iw;
                    vec4_t vec_val = *reinterpret_cast<const vec4_t*>(&input[input_idx]);
                    max_val = max(max_val, max(max(vec_val.x, vec_val.y), max(vec_val.z, vec_val.w)));
                } else {
                    for (int k = 0; k < 4 && kw + k < d_kernel_params[6]; k++) {
                        const int curr_iw = iw + k * d_kernel_params[9];
                        if (curr_iw >= 0 && curr_iw < d_kernel_params[3]) {
                            const int input_idx = base_input_offset + ih_offset + curr_iw;
                            max_val = max(max_val, __ldg(&input[input_idx]));
                        }
                    }
                }
            }
        }
    }

    const int output_write_idx = b * d_kernel_params[11] + 
                                c * d_kernel_params[4] * d_kernel_params[5] + 
                                oh * d_kernel_params[5] + 
                                ow;
    output[output_write_idx] = max_val;
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

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>()
        );
    }));

    return output;
}