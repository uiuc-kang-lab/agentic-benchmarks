#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_kernel(
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
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.z;
    const int channel_idx = blockIdx.y;
    const int row = blockIdx.x / ((output_width + 31) / 32) * 32 + threadIdx.x / 32;
    const int col = (blockIdx.x % ((output_width + 31) / 32)) * 32 + threadIdx.x % 32;
    
    if (row < output_height && col < output_width) {
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            const int ih = row * stride - padding + kh * dilation;
            
            if (ih >= 0 && ih < input_height) {
                #pragma unroll
                for (int kw = 0; kw < kernel_size; kw++) {
                    const int iw = col * stride - padding + kw * dilation;
                    
                    if (iw >= 0 && iw < input_width) {
                        const int input_idx = batch_idx * (channels * input_height * input_width) +
                                            channel_idx * (input_height * input_width) +
                                            ih * input_width +
                                            iw;
                        max_val = max(max_val, input[input_idx]);
                    }
                }
            }
        }
        
        const int output_idx = batch_idx * (channels * output_height * output_width) +
                              channel_idx * (output_height * output_width) +
                              row * output_width +
                              col;
        output[output_idx] = max_val;
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

    const dim3 threads(32 * 32);
    const dim3 blocks(
        ((output_height + 31) / 32) * ((output_width + 31) / 32),
        channels,
        batch_size
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}