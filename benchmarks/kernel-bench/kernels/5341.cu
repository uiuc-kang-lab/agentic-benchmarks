#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Kernel that uses grid-stride loops to evenly distribute workload across threads and blocks
template <typename scalar_t>
__global__ void balanced_max_pool2d_kernel(
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
    const int total_elements = batch_size * channels * output_height * output_width;
    
    // Grid-stride loop: each thread processes multiple output elements if necessary
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
        // Determine the output coordinates
        int ow = idx % output_width;
        int oh = (idx / output_width) % output_height;
        int c  = (idx / (output_width * output_height)) % channels;
        int b  = idx / (output_width * output_height * channels);

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        // Calculate base offset for input corresponding to batch and channel
        int input_offset = b * channels * input_height * input_width + c * input_height * input_width;
        
        // Iterate over the pooling window
        for (int kh = 0; kh < kernel_size; kh++) {
            int ih = oh * stride - padding + kh * dilation;
            if (ih < 0 || ih >= input_height) continue;
            int row_offset = input_offset + ih * input_width;
            for (int kw = 0; kw < kernel_size; kw++) {
                int iw = ow * stride - padding + kw * dilation;
                if (iw < 0 || iw >= input_width) continue;
                int input_idx = row_offset + iw;
                max_val = max(max_val, __ldg(&input[input_idx]));
            }
        }
        
        output[idx] = max_val;
    }
}

// Host function wrapping the kernel launch
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

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());
    
    const int total_elements = batch_size * channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        balanced_max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with evenly distributed workload (CUDA)");
}
