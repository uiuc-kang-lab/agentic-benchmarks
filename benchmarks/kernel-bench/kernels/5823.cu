#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// This kernel uses a grid-stride loop to distribute workload evenly among threads

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
    int total = batch_size * channels * output_height * output_width;
    
    // Grid-stride loop to ensure all threads process multiple output elements if needed
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
        // Decode flat index into (b, c, oh, ow) coordinates
        int ow = index % output_width;
        int tmp = index / output_width;
        int oh = tmp % output_height;
        tmp = tmp / output_height;
        int c = tmp % channels;
        int b = tmp / channels;
        
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        // Determine the top-left coordinate of the pooling window in the input
        int start_y = oh * stride - padding;
        int start_x = ow * stride - padding;
        
        // Loop over the pooling window
        #pragma unroll
        for (int kh = 0; kh < kernel_size; ++kh) {
            int ih = start_y + kh * dilation;
            if (ih < 0 || ih >= input_height) continue;
            #pragma unroll
            for (int kw = 0; kw < kernel_size; ++kw) {
                int iw = start_x + kw * dilation;
                if (iw < 0 || iw >= input_width) continue;
                
                int input_idx = b * (channels * input_height * input_width) +
                                c * (input_height * input_width) +
                                ih * input_width +
                                iw;
                max_val = max(max_val, input[input_idx]);
            }
        }
        
        output[index] = max_val;
    }
}


// Host function
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
    
    int total = batch_size * channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA) with even workload distribution via grid-stride loop");
}
