#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Inline device function for computing the minimum of two integers
__device__ inline int dmin(int a, int b) {
    return a < b ? a : b;
}

// Optimized CUDA kernel for 3D max pooling combining flat-index mapping for minimal warp divergence
// with __ldg for read-only cache optimized global memory accesses
template <typename scalar_t>
__global__ void max_pool3d_forward_optimized_kernel(
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
    const int total = batch_size * channels * output_d * output_h * output_w;
    if (idx >= total) return;

    // Compute the multi-dimensional output indices from the flat index
    const int w_out = idx % output_w;
    const int h_out = (idx / output_w) % output_h;
    const int d_out = (idx / (output_w * output_h)) % output_d;
    const int c = (idx / (output_w * output_h * output_d)) % channels;
    const int b = idx / (output_w * output_h * output_d * channels);

    // Define the top-left-front corner (starting indices) of the pooling window in the input tensor
    const int d_start = d_out * stride - padding;
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;

    // Precompute loop bounds to avoid unnecessary if-checks inside inner loops
    const int k_d_min = (d_start < 0) ? ((-d_start + dilation - 1) / dilation) : 0;
    const int valid_d_max = (input_d - d_start + dilation - 1) / dilation;
    const int k_d_max = dmin(kernel_size, valid_d_max);

    const int k_h_min = (h_start < 0) ? ((-h_start + dilation - 1) / dilation) : 0;
    const int valid_h_max = (input_h - h_start + dilation - 1) / dilation;
    const int k_h_max = dmin(kernel_size, valid_h_max);

    const int k_w_min = (w_start < 0) ? ((-w_start + dilation - 1) / dilation) : 0;
    const int valid_w_max = (input_w - w_start + dilation - 1) / dilation;
    const int k_w_max = dmin(kernel_size, valid_w_max);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_index = -1;

    // Iterate over the pooling window with loop unrolling to reduce overhead
    #pragma unroll
    for (int kd = k_d_min; kd < k_d_max; kd++) {
        const int d_in = d_start + kd * dilation;
        #pragma unroll
        for (int kh = k_h_min; kh < k_h_max; kh++) {
            const int h_in = h_start + kh * dilation;
            #pragma unroll
            for (int kw = k_w_min; kw < k_w_max; kw++) {
                const int w_in = w_start + kw * dilation;
                const int input_idx = (((b * channels + c) * input_d + d_in) * input_h + h_in) * input_w + w_in;
                // Use __ldg to load read-only data in a coalesced and cache-friendly manner
                const scalar_t val = __ldg(&input[input_idx]);
                if (val > max_val) {
                    max_val = val;
                    max_index = input_idx;
                }
            }
        }
    }

    output[idx] = max_val;
    if (indices != nullptr) {
        indices[idx] = max_index;
    }
}

// Host function to launch the optimized 3D max pooling kernel
torch::Tensor max_pool3d_cuda_forward_optimized(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool return_indices,
    bool ceil_mode) {

    const auto input_sizes = input.sizes();
    const int batch_size = input_sizes[0];
    const int channels = input_sizes[1];
    const int input_d = input_sizes[2];
    const int input_h = input_sizes[3];
    const int input_w = input_sizes[4];

    // Compute output dimensions
    float d_out_f = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1;
    float h_out_f = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1;
    float w_out_f = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1;

    int output_d = ceil_mode ? std::ceil(d_out_f) : std::floor(d_out_f);
    int output_h = ceil_mode ? std::ceil(h_out_f) : std::floor(h_out_f);
    int output_w = ceil_mode ? std::ceil(w_out_f) : std::floor(w_out_f);

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    torch::Tensor indices = return_indices ?
        torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kLong)) :
        torch::Tensor();

    const int total = batch_size * channels * output_d * output_h * output_w;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cuda_optimized", ([&] {
        max_pool3d_forward_optimized_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool3d_cuda_forward_optimized, "Optimized 3D Max Pool forward (CUDA) with __ldg and minimal divergence");
}
