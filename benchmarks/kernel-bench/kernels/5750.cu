#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* input,
    scalar_t* output,
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
    // Calculate position using block-wide coalesced indexing
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Early exit if out of bounds
    if (x >= output_width || y >= output_height) return;
    
    const int input_slice = input_height * input_width;
    const int output_slice = output_height * output_width;
    
    // Process all batches and channels for this spatial location
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
            
            // Calculate base indices for input and output
            const int output_idx = b * channels * output_slice + 
                                  c * output_slice + 
                                  y * output_width + x;
            const int input_batch_offset = b * channels * input_slice;
            const int input_channel_offset = c * input_slice;
            
            // Calculate input spatial location
            const int ih_start = y * stride - padding;
            const int iw_start = x * stride - padding;
            
            // Perform max pooling
            for (int kh = 0; kh < kernel_size; kh++) {
                const int ih = ih_start + kh * dilation;
                if (ih >= 0 && ih < input_height) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        const int iw = iw_start + kw * dilation;
                        if (iw >= 0 && iw < input_width) {
                            const int input_idx = input_batch_offset +
                                                 input_channel_offset +
                                                 ih * input_width + iw;
                            max_val = max(max_val, input[input_idx]);
                        }
                    }
                }
            }
            
            output[output_idx] = max_val;
        }
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

    // Use 2D grid and block configuration for better spatial locality
    dim3 threads(16, 16);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y
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