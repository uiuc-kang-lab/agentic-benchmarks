#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// This kernel assigns one warp per output element, and uses warp-level primitives
// (i.e., __shfl_down_sync) to perform the reduction across pooling window elements.
// Each thread in the warp loads one or more elements from the pooling window and
// then participates in a warp-level max reduction. This avoids using shared memory
// for small reductions and can improve performance.

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
    // Compute global thread ID
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // Each warp (32 threads) computes one output element
    int warp_id = global_thread_id / warpSize;
    int lane = threadIdx.x & (warpSize - 1);

    int total_outputs = batch_size * channels * output_height * output_width;
    if (warp_id >= total_outputs) return;

    // Decode warp_id into (b, c, oh, ow)
    int ow = warp_id % output_width;
    int tmp = warp_id / output_width;
    int oh = tmp % output_height;
    int tmp2 = tmp / output_height;
    int c = tmp2 % channels;
    int b = tmp2 / channels;

    // Calculate pooling window size
    int pool_size = kernel_size * kernel_size;
    // Compute base indices for the pooling window
    int base_ih = oh * stride - padding;
    int base_iw = ow * stride - padding;
    
    const scalar_t* input_channel = input + (b * channels + c) * input_height * input_width;
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();

    // Each thread in the warp loads one or more elements from the pooling window.
    // If the pooling window size is less than warpSize, some threads are idle.
    for (int idx = lane; idx < pool_size; idx += warpSize) {
        int kh = idx / kernel_size;
        int kw = idx % kernel_size;
        int ih = base_ih + kh * dilation;
        int iw = base_iw + kw * dilation;
        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
            local_max = max(local_max, input_channel[ih * input_width + iw]);
        }
    }

    // Perform warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        scalar_t other = __shfl_down_sync(mask, local_max, offset);
        local_max = max(local_max, other);
    }

    // The first lane writes the computed max to the output
    if (lane == 0) {
        output[warp_id] = local_max;
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
    const auto channels   = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width  = input.size(3);
    
    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    int total_outputs = batch_size * channels * output_height * output_width;
    // Each output element is computed by a warp (32 threads)
    int total_threads = total_outputs * warpSize;
    int threads = 256; // block size (must be a multiple of warpSize)
    int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_warp_reduce_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with warp-level reduction (CUDA)");
}
