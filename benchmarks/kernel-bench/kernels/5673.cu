#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// CUDA kernel using grid stride loops for handling large workloads
// and optimizing memory access patterns

template <typename scalar_t>
__global__ void max_pool2d_stride_loop_kernel(
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
    // total number of output elements
    const int total = batch_size * channels * output_height * output_width;
    
    // Use grid-stride loop to process all output elements
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int gridSize = blockDim.x * gridDim.x;
    
    while (index < total) {
        // Decompose flat index into 4D indices
        int ow = index % output_width;
        int oh = (index / output_width) % output_height;
        int c  = (index / (output_width * output_height)) % channels;
        int b  = index / (output_width * output_height * channels);

        // Initialize maximum value to -infinity
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        // For current output element, iterate over the kernel window
        // Compute offsets for the input tensor
        int input_batch_offset = b * channels * input_height * input_width;
        int input_channel_offset = c * input_height * input_width;
        
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = oh * stride - padding + kh * dilation;
                int iw = ow * stride - padding + kw * dilation;
                
                // Check boundaries
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    int input_idx = input_batch_offset + input_channel_offset + ih * input_width + iw;
                    scalar_t val = __ldg(&input[input_idx]);
                    max_val = (val > max_val) ? val : max_val;
                }
            }
        }

        output[index] = max_val;

        index += gridSize;
    }
}

// Host function to launch the CUDA kernel
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
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_stride_loop_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with stride loops (CUDA)");
}
