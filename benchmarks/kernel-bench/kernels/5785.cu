#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>


// Kernel using warp-level primitives for reduction
// Each warp computes one output element of the max pooling operation

template <typename scalar_t>
__global__ void max_pool2d_warp_reduce_kernel(
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
    // Each warp is assigned one output element
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int lane = threadIdx.x % warpSize;

    const int total_output_elements = batch_size * channels * output_height * output_width;
    if (warp_id >= total_output_elements) return;

    // Decode flattened output index
    const int out_index = warp_id; // mapping one warp to one output element
    const int ow = out_index % output_width;
    const int oh = (out_index / output_width) % output_height;
    const int bc = out_index / (output_height * output_width);  // combined batch*channels
    const int b = bc / channels;
    const int c = bc % channels;

    // Compute the starting position of the pooling window in the input
    const int h_start = oh * stride - padding;
    const int w_start = ow * stride - padding;
    const int window_size = kernel_size * kernel_size;

    // Each lane processes a stripe of the pooling region
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();
    for (int i = lane; i < window_size; i += warpSize) {
        int kh = i / kernel_size;
        int kw = i % kernel_size;
        int ih = h_start + kh * dilation;
        int iw = w_start + kw * dilation;
        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
            int input_idx = ((b * channels + c) * input_height + ih) * input_width + iw;
            thread_max = max(thread_max, input[input_idx]);
        }
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        scalar_t other = __shfl_down_sync(mask, thread_max, offset);
        thread_max = max(thread_max, other);
    }

    if (lane == 0) {
        output[out_index] = thread_max;
    }
}


// Wrapper function

torch::Tensor max_pool2d_cuda_forward_warp_reduce(
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
    
    const int total_output_elements = batch_size * channels * output_height * output_width;
    // One warp (32 threads) per output element
    const int warp_size = 32;
    const int total_threads = total_output_elements * warp_size;
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_forward_warp_reduce", ([&] {
        max_pool2d_warp_reduce_kernel<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &max_pool2d_cuda_forward_warp_reduce, "Max Pool 2D forward with warp-level reduction (CUDA)");
}
