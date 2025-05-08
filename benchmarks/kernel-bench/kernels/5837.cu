#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

__device__ __forceinline__ int compute_start_position(int out_idx, int stride, int padding) {
    return out_idx * stride - padding;
}

__device__ __forceinline__ int compute_pool_bounds(int start, int input_size, int kernel_size, int dilation, bool is_start) {
    if (is_start) {
        return (start < 0) ? ((-start + dilation - 1) / dilation) : 0;
    } else {
        int valid_max = (input_size - start + dilation - 1) / dilation;
        return min(kernel_size, valid_max);
    }
}

__device__ __forceinline__ void compute_output_indices(
    int idx, int output_w, int output_h, int output_d, int channels,
    int& w_out, int& h_out, int& d_out, int& c, int& b) {
    w_out = idx % output_w;
    h_out = (idx / output_w) % output_h;
    d_out = (idx / (output_w * output_h)) % output_d;
    c = (idx / (output_w * output_h * output_d)) % channels;
    b = idx / (output_w * output_h * output_d * channels);
}

__device__ __forceinline__ int compute_input_index(
    int b, int c, int d, int h, int w,
    int channels, int input_d, int input_h, int input_w) {
    return (((b * channels + c) * input_d + d) * input_h + h) * input_w + w;
}

// Helper function to perform warp-level reduction to find maximum value
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Optimized CUDA kernel utilizing warp-level primitives
template <typename scalar_t>
__global__ void max_pool3d_forward_warp_kernel(
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

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_d * output_h * output_w) return;

    int w_out, h_out, d_out, c, b;
    compute_output_indices(idx, output_w, output_h, output_d, channels, w_out, h_out, d_out, c, b);

    const int d_start = compute_start_position(d_out, stride, padding);
    const int h_start = compute_start_position(h_out, stride, padding);
    const int w_start = compute_start_position(w_out, stride, padding);

    const int k_d_start = compute_pool_bounds(d_start, input_d, kernel_size, dilation, true);
    const int k_d_end = compute_pool_bounds(d_start, input_d, kernel_size, dilation, false);
    const int k_h_start = compute_pool_bounds(h_start, input_h, kernel_size, dilation, true);
    const int k_h_end = compute_pool_bounds(h_start, input_h, kernel_size, dilation, false);
    const int k_w_start = compute_pool_bounds(w_start, input_w, kernel_size, dilation, true);
    const int k_w_end = compute_pool_bounds(w_start, input_w, kernel_size, dilation, false);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_index = -1;

    for (int k_d = k_d_start; k_d < k_d_end; k_d++) {
        const int d_in = d_start + k_d * dilation;
        for (int k_h = k_h_start; k_h < k_h_end; k_h++) {
            const int h_in = h_start + k_h * dilation;
            for (int k_w = k_w_start; k_w < k_w_end; k_w++) {
                const int w_in = w_start + k_w * dilation;
                const int input_idx = compute_input_index(b, c, d_in, h_in, w_in,
                                                        channels, input_d, input_h, input_w);
                const scalar_t val = input[input_idx];
                max_val = max(max_val, val);
                if (val == max_val) {
                    max_index = input_idx;
                }
            }
        }
    }

    // Perform warp-level reduction to find overall max value
    max_val = warpReduceMax(max_val);

    // Update output and indices
    output[idx] = max_val;
    if (indices != nullptr) {
        indices[idx] = max_index;
    }
}

// Host function for CUDA max pooling
torch::Tensor max_pool3d_cuda_forward_warp(
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

    const int threads = 256;
    const int blocks = (batch_size * channels * output_d * output_h * output_w + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cuda_warp", ([&] {
        max_pool3d_forward_warp_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool3d_cuda_forward_warp, "Max Pool 3D forward with warp reduction (CUDA)");
}
