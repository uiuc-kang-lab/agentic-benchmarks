#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Inline device function to compute minimum of two integers
__device__ inline int dmin(int a, int b) {
    return a < b ? a : b;
}

// Combined CUDA kernel for 3D max pooling that minimizes warp divergence
// by precomputing valid loop bounds and unrolling inner loops

template <typename scalar_t>
__global__ void max_pool3d_forward_combined_kernel(
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

    // Compute output indices
    const int w_out = idx % output_w;
    const int h_out = (idx / output_w) % output_h;
    const int d_out = (idx / (output_w * output_h)) % output_d;
    const int c = (idx / (output_w * output_h * output_d)) % channels;
    const int b = idx / (output_w * output_h * output_d * channels);

    // Compute starting positions in input tensor
    const int d_start = d_out * stride - padding;
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;

    // Precompute valid loop bounds for each dimension to avoid branch divergence
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

    // Loop over the valid pooling window with unrolling to reduce loop overhead
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
                const scalar_t val = input[input_idx];
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

// Host function that sets up the kernel launch and computes output dimensions

torch::Tensor max_pool3d_cuda_forward_combined(
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

    // Calculate output dimensions based on ceil_mode
    int output_d = ceil_mode ?
        std::ceil((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        std::floor((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);
    int output_h = ceil_mode ?
        std::ceil((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        std::floor((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);
    int output_w = ceil_mode ?
        std::ceil((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        std::floor((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = return_indices ?
        torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kLong)) :
        torch::Tensor();

    const int threads = 256;
    const int total = batch_size * channels * output_d * output_h * output_w;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cuda_combined", ([&] {
        max_pool3d_forward_combined_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool3d_cuda_forward_combined, "Max Pool 3D forward combined optimized (CUDA)");
}
