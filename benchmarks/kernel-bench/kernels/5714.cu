#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Inline device function for maximum operation
template <typename scalar_t>
__device__ inline scalar_t max_op(scalar_t a, scalar_t b) {
    return a > b ? a : b;
}

// Each block computes one output element using cooperative reduction.
// Threads in the block load portions of the pooling window into registers,
// then use shared memory and warp-level shuffle primitives to reduce the partial
// results into the final maximum value.

template <typename scalar_t>
__global__ void max_pool2d_shared_warp_kernel(
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
    // Each block is responsible for one output element
    int out_index = blockIdx.x;
    int total_outputs = batch_size * channels * output_height * output_width;
    if (out_index >= total_outputs) return;

    // Map linear out_index to (b, c, oh, ow)
    int ow = out_index % output_width;
    int oh = (out_index / output_width) % output_height;
    int c  = (out_index / (output_width * output_height)) % channels;
    int b  = out_index / (output_width * output_height * channels);

    int pool_size = kernel_size * kernel_size;
    // Initialize partial maximum to -infinity
    scalar_t partial_max = -std::numeric_limits<scalar_t>::infinity();

    // Each thread processes a subset of the pooling window
    for (int i = threadIdx.x; i < pool_size; i += blockDim.x) {
        int kh = i / kernel_size;
        int kw = i % kernel_size;
        int ih = oh * stride - padding + kh * dilation;
        int iw = ow * stride - padding + kw * dilation;

        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
            int input_idx = b * (channels * input_height * input_width) +
                            c * (input_height * input_width) +
                            ih * input_width + iw;
            partial_max = max_op(partial_max, input[input_idx]);
        }
    }

    // Allocate shared memory for intra-block reduction
    extern __shared__ scalar_t sdata[];
    sdata[threadIdx.x] = partial_max;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = max_op(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Use warp-level primitives for final reduction
    if (threadIdx.x < 32) {
        scalar_t val = sdata[threadIdx.x];
        val = max_op(val, __shfl_down_sync(0xffffffff, val, 16));
        val = max_op(val, __shfl_down_sync(0xffffffff, val, 8));
        val = max_op(val, __shfl_down_sync(0xffffffff, val, 4));
        val = max_op(val, __shfl_down_sync(0xffffffff, val, 2));
        val = max_op(val, __shfl_down_sync(0xffffffff, val, 1));
        if (threadIdx.x == 0) {
            output[out_index] = val;
        }
    }
}


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

    int total_outputs = batch_size * channels * output_height * output_width;
    // Choose a block size that allows multiple threads to reduce the pooling window
    const int threads = 64;
    const int blocks = total_outputs;  // one block per output element

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_shared_warp_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
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
