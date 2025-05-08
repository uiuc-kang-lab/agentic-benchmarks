#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// CUDA kernel where each warp computes one output element using warp-level reduction

template <typename scalar_t>
__global__ void max_pool2d_warp_kernel(
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
    // Each warp computes one output element.
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / warpSize;  // one warp per output element
    int lane = threadIdx.x % warpSize;

    int total_output = batch_size * channels * output_height * output_width;
    if (warp_id >= total_output) return;

    // Decode warp_id into (b, c, oh, ow)
    int ow = warp_id % output_width;
    int temp = warp_id / output_width;
    int oh = temp % output_height;
    temp = temp / output_height;
    int c = temp % channels;
    int b = temp / channels;

    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();

    // Calculate the total number of elements in the pooling window
    int pool_area = kernel_size * kernel_size;

    // Each thread in the warp processes a subset of pooling window elements
    for (int i = lane; i < pool_area; i += warpSize) {
        int kh = i / kernel_size;
        int kw = i % kernel_size;
        int ih = oh * stride - padding + kh * dilation;
        int iw = ow * stride - padding + kw * dilation;

        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
            int input_idx = b * (channels * input_height * input_width) +
                            c * (input_height * input_width) +
                            ih * input_width + iw;
            scalar_t val = input[input_idx];
            local_max = (val > local_max) ? val : local_max;
        }
    }

    // Perform warp-level reduction using __shfl_down_sync to compute max
    // Use full mask for active lanes
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        scalar_t other = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = (local_max > other) ? local_max : other;
    }

    // The first lane writes the result
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
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Each output element is computed by one warp
    int total_output = batch_size * channels * output_height * output_width;
    // Total threads required = total_output * warpSize
    int threads_per_block = 256;
    int total_threads = total_output * warpSize;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_warp_kernel<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA) with warp-level primitives");
}
