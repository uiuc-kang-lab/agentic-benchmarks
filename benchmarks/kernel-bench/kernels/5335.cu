#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// This kernel assigns one block per output element. Threads within the block collaboratively process the pooling window using shared memory and warp-level reduction via __shfl_down_sync().

template <typename scalar_t>
__global__ void max_pool2d_kernel_shared_warp(
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
    // Each block computes one output element
    const int out_idx = blockIdx.x;
    if (out_idx >= batch_size * channels * output_height * output_width) return;

    // Decode the flat output index into (n, c, oh, ow)
    int n = out_idx / (channels * output_height * output_width);
    int rem = out_idx % (channels * output_height * output_width);
    int c = rem / (output_height * output_width);
    int rem2 = rem % (output_height * output_width);
    int oh = rem2 / output_width;
    int ow = rem2 % output_width;

    int window_size = kernel_size * kernel_size;
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();

    // Each thread in the block processes part of the pooling window
    for (int idx = threadIdx.x; idx < window_size; idx += blockDim.x) {
        int kh = idx / kernel_size;
        int kw = idx % kernel_size;
        int ih = oh * stride - padding + kh * dilation;
        int iw = ow * stride - padding + kw * dilation;
        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
            int input_index = n * (channels * input_height * input_width) +
                              c * (input_height * input_width) +
                              ih * input_width + iw;
            scalar_t val = __ldg(&input[input_index]);
            thread_max = max(thread_max, val);
        }
    }

    // Intra-warp reduction using warp-level intrinsics
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        scalar_t other = __shfl_down_sync(mask, thread_max, offset);
        thread_max = max(thread_max, other);
    }

    // Shared memory reduction among warps in the block
    __shared__ scalar_t shared_mem[32]; // enough for up to 32 warps per block
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;  // threadIdx.x / 32

    if (lane == 0) {
        shared_mem[warpId] = thread_max;
    }
    __syncthreads();

    // Final reduction using the first warp
    scalar_t block_max = -std::numeric_limits<scalar_t>::infinity();
    if (threadIdx.x < (blockDim.x + 31) / 32) {
        block_max = shared_mem[lane];
    }
    if (warpId == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            scalar_t other = __shfl_down_sync(mask, block_max, offset);
            block_max = max(block_max, other);
        }
        if (lane == 0) {
            output[out_idx] = block_max;
        }
    }
}

// The CUDA forward function maps one block to each output element

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

    // Each block computes one output element
    const int total_outputs = batch_size * channels * output_height * output_width;
    const int threads = 128; // number of threads per block
    const int blocks = total_outputs;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel_shared_warp<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with shared memory and warp-level reduction (CUDA)");
}
