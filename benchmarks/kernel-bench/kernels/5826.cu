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
    const int total_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    
    // Each thread processes multiple elements with stride
    for (int output_idx = tid; output_idx < total_elements; output_idx += total_threads) {
        // Calculate position in output tensor
        const int ow = output_idx % output_width;
        int tmp = output_idx / output_width;
        const int oh = tmp % output_height;
        tmp /= output_height;
        const int c = tmp % channels;
        const int b = tmp / channels;

        // Calculate input boundaries for this output position
        const int in_y_start = oh * stride - padding;
        const int in_x_start = ow * stride - padding;
        
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        // Pre-calculate valid kernel bounds
        const int kh_start = max(0, (-in_y_start + dilation - 1) / dilation);
        const int kw_start = max(0, (-in_x_start + dilation - 1) / dilation);
        const int kh_end = min(kernel_size, (input_height - in_y_start + dilation - 1) / dilation);
        const int kw_end = min(kernel_size, (input_width - in_x_start + dilation - 1) / dilation);

        // Base offset for current batch and channel
        const int base_offset = b * (channels * input_height * input_width) +
                              c * (input_height * input_width);

        // Process only valid positions within the kernel
        #pragma unroll 4
        for (int kh = kh_start; kh < kh_end; ++kh) {
            const int ih = in_y_start + kh * dilation;
            const int row_offset = ih * input_width;
            
            #pragma unroll 4
            for (int kw = kw_start; kw < kw_end; ++kw) {
                const int iw = in_x_start + kw * dilation;
                const int input_idx = base_offset + row_offset + iw;
                max_val = max(max_val, input[input_idx]);
            }
        }
        
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

    const int total_elements = batch_size * channels * output_height * output_width;
    
    // Optimize thread and block configuration for H100
    const int threads_per_block = 256;
    const int max_blocks = 65535;
    const int min_elements_per_thread = 4;
    
    // Calculate optimal number of blocks
    int num_blocks = min(
        max_blocks,
        (total_elements + threads_per_block * min_elements_per_thread - 1) / 
        (threads_per_block * min_elements_per_thread)
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
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
            total_elements
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}