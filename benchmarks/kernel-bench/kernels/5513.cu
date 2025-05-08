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
    const int dilation,
    const int input_stride_batch,
    const int input_stride_channel,
    const int output_stride_batch,
    const int output_stride_channel
) {
    // Map thread to output position ensuring coalesced access
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_width = output_width * channels;
    
    if (tid >= batch_size * output_height * total_width) return;

    const int w_c = tid % total_width;
    const int h = (tid / total_width) % output_height;
    const int b = tid / (output_height * total_width);
    
    const int c = w_c / output_width;
    const int w = w_c % output_width;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    const int input_batch_offset = b * input_stride_batch;
    const int input_channel_offset = c * input_stride_channel;
    
    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        const int ih = h * stride - padding + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            const int input_h_offset = ih * input_width;
            
            #pragma unroll
            for (int kw = 0; kw < kernel_size; kw++) {
                const int iw = w * stride - padding + kw * dilation;
                if (iw >= 0 && iw < input_width) {
                    const int input_idx = input_batch_offset + 
                                        input_channel_offset + 
                                        input_h_offset + 
                                        iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    }

    const int output_idx = b * output_stride_batch +
                          c * output_stride_channel +
                          h * output_width +
                          w;
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

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Calculate stride values for efficient memory access
    const int input_stride_batch = channels * input_height * input_width;
    const int input_stride_channel = input_height * input_width;
    const int output_stride_batch = channels * output_height * output_width;
    const int output_stride_channel = output_height * output_width;

    const int threads = 256;
    const int total_elements = batch_size * output_height * output_width * channels;
    const int blocks = (total_elements + threads - 1) / threads;

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
            dilation,
            input_stride_batch,
            input_stride_channel,
            output_stride_batch,
            output_stride_channel
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}