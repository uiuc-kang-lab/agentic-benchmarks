#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Warp-level max pooling kernel using __shfl_down_sync for reduction.
// Each warp computes one output element.

template <typename scalar_t>
__global__ void warp_max_pool2d_kernel(
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
    // Each warp computes one output pixel. Compute global thread id.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Warp id is the index of the output element this warp will compute.
    int warpId = tid / 32;
    int lane = tid % 32;

    // Total number of output pixels
    int total_outputs = batch_size * channels * output_height * output_width;
    if (warpId >= total_outputs) return;

    // Reconstruct (b, c, oh, ow) from warpId
    int ow = warpId % output_width;
    int tmp = warpId / output_width;
    int oh = tmp % output_height;
    tmp = tmp / output_height;
    int c = tmp % channels;
    int b = tmp / channels;

    // Compute the starting index in the input for this (b, c) channel
    int input_channel_offset = b * channels * input_height * input_width + c * input_height * input_width;
    
    // Compute top-left index for the pooling window
    int ih_start = oh * stride - padding;
    int iw_start = ow * stride - padding;

    // Total number of elements in the pooling window
    int num_elements = kernel_size * kernel_size;
    // Each thread in the warp loads one element if its lane id is within the pooling window size.
    // For lanes >= num_elements, assign -infinity.
    float cur_val = -INFINITY;
    if (lane < num_elements) {
        int kh = lane / kernel_size;
        int kw = lane % kernel_size;
        int in_h = ih_start + kh * dilation;
        int in_w = iw_start + kw * dilation;
        if (in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width) {
            int input_index = input_channel_offset + in_h * input_width + in_w;
            cur_val = __ldg(&input[input_index]);
        }
    }

    // Warp-level reduction using __shfl_down_sync to compute maximum
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(mask, cur_val, offset);
        cur_val = fmaxf(cur_val, other);
    }

    // Lane 0 writes the result
    if (lane == 0) {
        output[warpId] = cur_val;
    }
}


// Host function to launch the warp-level max pooling kernel
// Each warp (32 threads) computes one output pixel

torch::Tensor warp_max_pool2d_cuda_forward(
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

    // Total number of output pixels, each computed by one warp (32 threads)
    int total_outputs = batch_size * channels * output_height * output_width;
    int total_threads = total_outputs * 32; // 32 threads per warp
    int threads_per_block = 128; // e.g., 4 warps per block
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "warp_max_pool2d_cuda_forward", ([&] {
        warp_max_pool2d_kernel<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &warp_max_pool2d_cuda_forward, "Max Pool 2D forward using warp-level primitives (CUDA)");
}
