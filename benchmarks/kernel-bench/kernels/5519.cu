#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Size of warp in CUDA
constexpr int WARP_SIZE = 32;
// Number of elements per thread
constexpr int ELEMENTS_PER_THREAD = 4;

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
    // Calculate global thread index
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    
    // Total number of output elements
    const int total_elements = batch_size * channels * output_height * output_width;
    const int elements_per_warp = WARP_SIZE * ELEMENTS_PER_THREAD;
    
    // Each warp processes a contiguous chunk of memory
    const int warp_offset = warp_id * elements_per_warp;
    
    // Process multiple elements per thread
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int element_idx = warp_offset + lane_id + i * WARP_SIZE;
        if (element_idx >= total_elements) return;
        
        // Convert linear index to n,c,h,w coordinates
        const int w = element_idx % output_width;
        const int h = (element_idx / output_width) % output_height;
        const int c = (element_idx / (output_width * output_height)) % channels;
        const int b = element_idx / (output_width * output_height * channels);
        
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        // Input base offset for current batch and channel
        const int input_batch_offset = b * channels * input_height * input_width;
        const int input_channel_offset = c * input_height * input_width;
        
        // Compute max value for current output position
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
        
        // Write output with coalesced access pattern
        output[element_idx] = max_val;
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
    const int threads_per_block = 128; // Multiple of WARP_SIZE
    const int elements_per_block = threads_per_block * ELEMENTS_PER_THREAD;
    const int blocks = (total_elements + elements_per_block - 1) / elements_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads_per_block>>>(
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