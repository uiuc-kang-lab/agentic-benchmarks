#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Define the number of threads per block for the reduction kernel.
#define THREADS_PER_BLOCK 128

// Kernel: Each block computes one output element by reducing over the corresponding pooling window.
// The reduction is performed in two stages: intra-warp reduction using warp-level primitives and
// final reduction among warp winners using shared memory.

template <typename scalar_t>
__global__ void max_pool3d_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int channels,
    const int input_d, const int input_h, const int input_w,
    const int output_d, const int output_h, const int output_w,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {

    // Each block computes one output element. The grid is sized to total output elements.
    int out_index = blockIdx.x;
    int total_out = batch_size * channels * output_d * output_h * output_w;
    if (out_index >= total_out) return;

    // Decode flat out_index into indices: b, c, d_out, h_out, w_out
    int w_out = out_index % output_w;
    int tmp = out_index / output_w;
    int h_out = tmp % output_h;
    tmp = tmp / output_h;
    int d_out = tmp % output_d;
    tmp = tmp / output_d;
    int c = tmp % channels;
    int b = tmp / channels;

    // Compute starting indices in the input volume for the pooling window
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    // Total number of elements in the pooling window
    int pool_size = kernel_size * kernel_size * kernel_size;

    // Each thread in the block processes a subset of the pooling window elements
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    int local_max_idx = -1;

    for (int i = threadIdx.x; i < pool_size; i += blockDim.x) {
        int kd = i / (kernel_size * kernel_size);
        int rem = i % (kernel_size * kernel_size);
        int kh = rem / kernel_size;
        int kw = rem % kernel_size;

        int d_in = d_start + kd * dilation;
        int h_in = h_start + kh * dilation;
        int w_in = w_start + kw * dilation;

        scalar_t val = -std::numeric_limits<scalar_t>::infinity();
        int idx = -1;
        if (d_in >= 0 && d_in < input_d &&
            h_in >= 0 && h_in < input_h &&
            w_in >= 0 && w_in < input_w) {
            idx = ((b * channels + c) * input_d + d_in) * input_h * input_w + (h_in * input_w + w_in);
            val = input[idx];
        }

        if (val > local_max) {
            local_max = val;
            local_max_idx = idx;
        }
    }

    // Intra-warp reduction using warp-level primitives
    const unsigned int warpSize = 32;
    unsigned int lane = threadIdx.x & (warpSize - 1);
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        scalar_t other_val = __shfl_down_sync(0xffffffff, local_max, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_max_idx, offset);
        if (other_val > local_max) {
            local_max = other_val;
            local_max_idx = other_idx;
        }
    }

    // Allocate shared memory to store each warp's result
    __shared__ scalar_t s_val[THREADS_PER_BLOCK / 32];
    __shared__ int s_idx[THREADS_PER_BLOCK / 32];
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) {
        s_val[warp_id] = local_max;
        s_idx[warp_id] = local_max_idx;
    }
    __syncthreads();

    // Final reduction: thread 0 in the block reduces the warp winners
    if (threadIdx.x == 0) {
        int numWarps = blockDim.x / warpSize;
        scalar_t final_max = s_val[0];
        int final_idx = s_idx[0];
        for (int i = 1; i < numWarps; i++) {
            if (s_val[i] > final_max) {
                final_max = s_val[i];
                final_idx = s_idx[i];
            }
        }
        output[out_index] = final_max;
        if (indices != nullptr) {
            indices[out_index] = final_idx;
        }
    }
}


torch::Tensor max_pool3d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool return_indices,
    bool ceil_mode) {

    auto input_sizes = input.sizes();
    const int batch_size = input_sizes[0];
    const int channels = input_sizes[1];
    const int input_d = input_sizes[2];
    const int input_h = input_sizes[3];
    const int input_w = input_sizes[4];

    const int output_d = ceil_mode ?
        ceil((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        floor((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);
    const int output_h = ceil_mode ?
        ceil((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        floor((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);
    const int output_w = ceil_mode ?
        ceil((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        floor((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = return_indices ?
        torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kLong)) :
        torch::Tensor();

    int total_output_elements = batch_size * channels * output_d * output_h * output_w;
    dim3 blocks(total_output_elements);
    dim3 threads(THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cuda", ([&] {
        max_pool3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            return_indices ? indices.data_ptr<int64_t>() : nullptr,
            batch_size, channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            kernel_size, stride, padding, dilation);
    }));

    if (return_indices) {
        return torch::stack({output, indices}, 0);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool3d_cuda_forward, "3D Max Pooling forward (CUDA) with shared memory and warp-level reduction");
}
