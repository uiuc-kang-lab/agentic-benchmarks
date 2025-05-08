#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int const_params[4];

template <typename scalar_t>
__global__ void max_pool2d_aligned_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * channels * output_height * output_width;
    if (tid >= total_elements) return;

    const int kernel_size = const_params[0];
    const int stride = const_params[1];
    const int padding = const_params[2];
    const int dilation = const_params[3];

    // Calculate position
    const int ow = tid % output_width;
    const int oh = (tid / output_width) % output_height;
    const int c = (tid / (output_width * output_height)) % channels;
    const int b = tid / (output_width * output_height * channels);

    // Pre-calculate base offset for input
    const int input_batch_offset = b * channels * input_height * input_width;
    const int input_channel_offset = c * input_height * input_width;
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Align memory access to 128-bit boundaries when possible
    const int aligned_iw_start = (ow * stride - padding) & ~3;
    const int aligned_iw_end = ((ow * stride - padding + kernel_size * dilation) + 3) & ~3;

    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        const int ih = oh * stride - padding + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            const int row_offset = ih * input_width;
            
            for (int aligned_iw = aligned_iw_start; aligned_iw < aligned_iw_end; aligned_iw += 4) {
                if (aligned_iw >= 0 && aligned_iw + 3 < input_width) {
                    // Use vector load for aligned access
                    float4 input_vec = *reinterpret_cast<const float4*>(__ldg(&input[
                        input_batch_offset + 
                        input_channel_offset + 
                        row_offset + 
                        aligned_iw
                    ]));
                    
                    if (aligned_iw >= ow * stride - padding && 
                        aligned_iw < ow * stride - padding + kernel_size * dilation) {
                        max_val = max(max_val, input_vec.x);
                        max_val = max(max_val, input_vec.y);
                        max_val = max(max_val, input_vec.z);
                        max_val = max(max_val, input_vec.w);
                    }
                } else {
                    // Handle boundary cases with scalar loads
                    for (int i = 0; i < 4; i++) {
                        const int iw = aligned_iw + i;
                        if (iw >= 0 && iw < input_width &&
                            iw >= ow * stride - padding && 
                            iw < ow * stride - padding + kernel_size * dilation) {
                            const int input_idx = input_batch_offset + 
                                                input_channel_offset + 
                                                row_offset + 
                                                iw;
                            max_val = max(max_val, __ldg(&input[input_idx]));
                        }
                    }
                }
            }
        }
    }

    output[tid] = max_val;
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

    const int params[4] = {kernel_size, stride, padding, dilation};
    cudaMemcpyToSymbol(const_params, params, sizeof(int) * 4);

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_aligned_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}