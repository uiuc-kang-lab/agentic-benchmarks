#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Optimized and synchronized 3D max pooling kernel
// Utilizes shared memory for caching input data and minimizes the use of __syncthreads()

// Define shared memory size based on block
#define BLOCK_SIZE 8

// Kernel with shared memory and synchronization optimizations
// Precomputes valid bounds and minimizes thread sync with selective use of __syncthreads()
template <typename scalar_t>
__global__ void max_pool3d_forward_kernel_sync_optimized(
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

    extern __shared__ scalar_t shared_input[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * channels * output_d * output_h * output_w;
    if (idx >= total) return;

    // Compute output indices (w, h, d, channel, batch) inline
    int w_out = idx % output_w;
    int h_out = (idx / output_w) % output_h;
    int d_out = (idx / (output_w * output_h)) % output_d;
    int c = (idx / (output_w * output_h * output_d)) % channels;
    int b = idx / (output_w * output_h * output_d * channels);

    // Compute the start positions
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    // Precompute and check valid loop bounds to minimize shared memory
    int k_d_min = (d_start < 0) ? ((-d_start + dilation - 1) / dilation) : 0;
    int k_d_max = min(kernel_size, (input_d - d_start + dilation - 1) / dilation);

    int k_h_min = (h_start < 0) ? ((-h_start + dilation - 1) / dilation) : 0;
    int k_h_max = min(kernel_size, (input_h - h_start + dilation - 1) / dilation);

    int k_w_min = (w_start < 0) ? ((-w_start + dilation - 1) / dilation) : 0;
    int k_w_max = min(kernel_size, (input_w - w_start + dilation - 1) / dilation);

    // Load input via shared memory
    int shared_offset = threadIdx.x;
    for (int kd = k_d_min; kd < k_d_max; ++kd) {
        for (int kh = k_h_min; kh < k_h_max; ++kh) {
            for (int kw = k_w_min; kw < k_w_max; ++kw) {
                if (shared_offset < BLOCK_SIZE && d_start >= 0 && h_start >= 0 && w_start >= 0) {
                    int d_in = d_start + kd * dilation;
                    int h_in = h_start + kh * dilation;
                    int w_in = w_start + kw * dilation;

                    if (d_in < input_d && h_in < input_h && w_in < input_w) {
                        int input_idx = (((b * channels + c) * input_d + d_in) * input_h + h_in) * input_w + w_in;
                        shared_input[shared_offset] = input[input_idx];
                    }
                }
                __syncthreads();
            }
        }
    }

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_index = -1;

    // Iterate over the shared memory to find the max
    for (int kd = k_d_min; kd < k_d_max; ++kd) {
        int d_in = d_start + kd * dilation;
        for (int kh = k_h_min; kh < k_h_max; ++kh) {
            int h_in = h_start + kh * dilation;
            for (int kw = k_w_min; kw < k_w_max; ++kw) {
                int w_in = w_start + kw * dilation;

                int shared_idx = (((b * channels + c) * kernel_size + kd) * kernel_size + kh) * kernel_size + kw;
                scalar_t val = shared_input[shared_idx];

                if (val > max_val) {
                    max_val = val;
                    max_index = kd * (kernel_size * kernel_size) + kh * kernel_size + kw;
                }
            }
        }
    }

    output[idx] = max_val;
    if (indices != nullptr) {
        indices[idx] = max_index;
    }
}

// Host function to prepare and launch the CUDA kernel

torch::Tensor max_pool3d_cuda_forward_sync(
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

    // Calculate output dimensions
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

    const int threads = 256;
    const int total = batch_size * channels * output_d * output_h * output_w;
    const int blocks = (total + threads - 1) / threads;

    size_t shared_mem_size = BLOCK_SIZE * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cuda_sync", ([&] {
        max_pool3d_forward_kernel_sync_optimized<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            return_indices ? indices.data_ptr<int64_t>() : nullptr,
            batch_size,
            channels,
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
    m.def("forward", &max_pool3d_cuda_forward_sync, "Optimized Max Pool 3D forward with sync (CUDA)");
}
