#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void max_pool2d_balanced_kernel(
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
    // Use 2D block configuration for better work distribution
    const int tile_w = 8;
    const int tile_h = 8;
    
    __shared__ scalar_t shared_input[tile_h + 2][tile_w + 2];
    
    const int batch_idx = blockIdx.z / channels;
    const int channel_idx = blockIdx.z % channels;
    const int out_y = blockIdx.y * tile_h + threadIdx.y;
    const int out_x = blockIdx.x * tile_w + threadIdx.x;
    
    if (out_y >= output_height || out_x >= output_width) return;
    
    const int input_batch_offset = batch_idx * channels * input_height * input_width;
    const int input_channel_offset = channel_idx * input_height * input_width;
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    if (kernel_size == 2) {
        #pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < 2; ++kw) {
                const int ih = out_y * stride - padding + kh * dilation;
                const int iw = out_x * stride - padding + kw * dilation;
                
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    const int input_idx = input_batch_offset + input_channel_offset + 
                                        ih * input_width + iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    } else {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int ih = out_y * stride - padding + kh * dilation;
                const int iw = out_x * stride - padding + kw * dilation;
                
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    const int input_idx = input_batch_offset + input_channel_offset + 
                                        ih * input_width + iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    }
    
    const int output_idx = (batch_idx * channels + channel_idx) * output_height * output_width +
                          out_y * output_width + out_x;
    output[output_idx] = max_val;
}

torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    
    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());
    
    // Configure 2D block and grid dimensions for better work distribution
    const dim3 threads(8, 8);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_forward", ([&] {
        max_pool2d_balanced_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D balanced forward (CUDA)");
}