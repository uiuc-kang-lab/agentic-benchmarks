#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_coalesced_kernel_shared(
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
    // Map threads to process elements within the same row for better coalescing
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_count = gridDim.x * blockDim.x;
    
    // Process multiple output elements per thread if necessary
    const int total_elements = batch_size * channels * output_height * output_width;
    
    #pragma unroll 1
    for (int idx = tid; idx < total_elements; idx += thread_count) {
        // Calculate output coordinates ensuring coalesced access
        const int ow = idx % output_width;
        const int oh = (idx / output_width) % output_height;
        const int c = (idx / (output_width * output_height)) % channels;
        const int b = idx / (output_width * output_height * channels);
        
        // Calculate base input offset for this thread
        const int input_batch_offset = b * channels * input_height * input_width;
        const int input_channel_offset = c * input_height * input_width;
        
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        // Calculate window boundaries
        const int h_start = oh * stride - padding;
        const int w_start = ow * stride - padding;
        const int h_end = min(h_start + kernel_size * dilation, input_height + padding);
        const int w_end = min(w_start + kernel_size * dilation, input_width + padding);
        
        #pragma unroll 4
        for (int ih = h_start; ih < h_end; ih += dilation) {
            if (ih >= 0 && ih < input_height) {
                const int input_h_offset = ih * input_width;
                
                #pragma unroll 4
                for (int iw = w_start; iw < w_end; iw += dilation) {
                    if (iw >= 0 && iw < input_width) {
                        const int input_idx = input_batch_offset + 
                                            input_channel_offset + 
                                            input_h_offset + 
                                            iw;
                        max_val = max(max_val, __ldg(&input[input_idx]));
                    }
                }
            }
        }
        
        output[idx] = max_val;
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

    // Adjust block size to ensure good occupancy and aligned memory access
    const int threads = 256;
    const int total_elements = batch_size * channels * output_height * output_width;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_coalesced_kernel<scalar_t><<<blocks, threads>>>(
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