/*
Optimized 3D Max Pooling Kernel
This kernel combines the dynamic bound computation from Kernel 1 with loop unrolling from Kernel 2. By computing the valid kernel bounds ahead of time, unnecessary iterations over out-of-bound regions are eliminated, and the use of #pragma unroll reduces loop overhead.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Combined kernel: computes valid iteration bounds and applies loop unrolling over the kernel dimensions
template <typename scalar_t>
__global__ void max_pool3d_forward_kernel_combined(
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

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * channels * output_d * output_h * output_w;
    if (idx >= total) return;

    // Compute output indices (w, h, d, channel, batch) inline
    int w_out = idx % output_w;
    int h_out = (idx / output_w) % output_h;
    int d_out = (idx / (output_w * output_h)) % output_d;
    int c = (idx / (output_w * output_h * output_d)) % channels;
    int b = idx / (output_w * output_h * output_d * channels);

    // Compute the starting positions in the input tensor
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    // Precompute valid kernel bounds to skip out-of-bound computations
    int k_d_start = (d_start < 0) ? ((-d_start + dilation - 1) / dilation) : 0;
    int k_d_end = min(kernel_size, (input_d - d_start + dilation - 1) / dilation);

    int k_h_start = (h_start < 0) ? ((-h_start + dilation - 1) / dilation) : 0;
    int k_h_end = min(kernel_size, (input_h - h_start + dilation - 1) / dilation);

    int k_w_start = (w_start < 0) ? ((-w_start + dilation - 1) / dilation) : 0;
    int k_w_end = min(kernel_size, (input_w - w_start + dilation - 1) / dilation);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_index = -1;

    // Loop over the valid kernel window with unrolling directives
    #pragma unroll
    for (int k_d = k_d_start; k_d < k_d_end; k_d++) {
        int d_in = d_start + k_d * dilation;
        #pragma unroll
        for (int k_h = k_h_start; k_h < k_h_end; k_h++) {
            int h_in = h_start + k_h * dilation;
            #pragma unroll
            for (int k_w = k_w_start; k_w < k_w_end; k_w++) {
                int w_in = w_start + k_w * dilation;
                int input_idx = (((b * channels + c) * input_d + d_in) * input_h + h_in) * input_w + w_in;
                scalar_t val = input[input_idx];
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

// Host function to prepare and launch the CUDA kernel

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

    // Calculate output dimensions
    float d_out_float = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / static_cast<float>(stride) + 1;
    float h_out_float = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / static_cast<float>(stride) + 1;
    float w_out_float = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / static_cast<float>(stride) + 1;

    const int output_d = ceil_mode ? ceil(d_out_float) : floor(d_out_float);
    const int output_h = ceil_mode ? ceil(h_out_float) : floor(h_out_float);
    const int output_w = ceil_mode ? ceil(w_out_float) : floor(w_out_float);

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = return_indices ?
        torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kLong)) :
        torch::Tensor();

    const int threads = 256;
    const int total = batch_size * channels * output_d * output_h * output_w;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cuda", ([&] {
        max_pool3d_forward_kernel_combined<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool3d_cuda_forward, "Optimized Max Pool 3D forward (combined, CUDA)");
}
