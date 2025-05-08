#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Kernel using grid-stride loop for even workload distribution
// and optimized block size for better load balancing

template <typename scalar_t>
__global__ void max_pool2d_kernel_optimized(
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
    int total_elements = batch_size * channels * output_height * output_width;
    // Use grid-stride loop to evenly distribute work
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total_elements;
         index += gridDim.x * blockDim.x) {

        int ow = index % output_width;
        int temp = index / output_width;
        int oh = temp % output_height;
        temp /= output_height;
        int c = temp % channels;
        int b = temp / channels;

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        int base_h = oh * stride - padding;
        int base_w = ow * stride - padding;
        
        for (int kh = 0; kh < kernel_size; kh++) {
            int ih = base_h + kh * dilation;
            if (ih < 0 || ih >= input_height) continue;
            for (int kw = 0; kw < kernel_size; kw++) {
                int iw = base_w + kw * dilation;
                if (iw < 0 || iw >= input_width) continue;
                int input_index = ((b * channels + c) * input_height + ih) * input_width + iw;
                max_val = max(max_val, input[input_index]);
            }
        }
        output[index] = max_val;
    }
}

// Host function to launch the CUDA kernel
torch::Tensor max_pool2d_cuda_forward_optimized(
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
    
    // Compute output dimensions
    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int total_elements = batch_size * channels * output_height * output_width;
    const int threads = 256;  // Increased thread count for better performance
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward_optimized", ([&] {
        max_pool2d_kernel_optimized<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward_optimized, "Max Pool 2D forward with optimized load balancing (CUDA)");
}
