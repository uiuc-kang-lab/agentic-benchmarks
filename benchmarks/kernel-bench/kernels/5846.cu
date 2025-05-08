#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

template <typename scalar_t>
__global__ void max_pool3d_forward_kernel_aligned(
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

    // Align thread index to 128-bit boundary (4 float elements)
    const int aligned_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int total_elements = batch_size * channels * output_d * output_h * output_w;

    // Process 4 elements per thread when possible
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int idx = aligned_idx + i;
        if (idx >= total_elements) continue;

        // Compute output indices
        const int w_out = idx % output_w;
        const int h_out = (idx / output_w) % output_h;
        const int d_out = (idx / (output_w * output_h)) % output_d;
        const int c = (idx / (output_w * output_h * output_d)) % channels;
        const int b = idx / (output_w * output_h * output_d * channels);

        // Compute input bounds
        const int d_start = d_out * stride - padding;
        const int h_start = h_out * stride - padding;
        const int w_start = w_out * stride - padding;

        // Precompute valid ranges
        const int k_d_start = (d_start < 0) ? ((-d_start + dilation - 1) / dilation) : 0;
        const int k_d_end = min(kernel_size, (input_d - d_start + dilation - 1) / dilation);

        const int k_h_start = (h_start < 0) ? ((-h_start + dilation - 1) / dilation) : 0;
        const int k_h_end = min(kernel_size, (input_h - h_start + dilation - 1) / dilation);

        const int k_w_start = (w_start < 0) ? ((-w_start + dilation - 1) / dilation) : 0;
        const int k_w_end = min(kernel_size, (input_w - w_start + dilation - 1) / dilation);

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        int max_index = -1;

        // Process pooling window with optimized memory access
        #pragma unroll
        for (int k_d = k_d_start; k_d < k_d_end; k_d++) {
            const int d_in = d_start + k_d * dilation;
            #pragma unroll
            for (int k_h = k_h_start; k_h < k_h_end; k_h++) {
                const int h_in = h_start + k_h * dilation;
                #pragma unroll
                for (int k_w = k_w_start; k_w < k_w_end; k_w++) {
                    const int w_in = w_start + k_w * dilation;
                    const int input_idx = (((b * channels + c) * input_d + d_in) * input_h + h_in) * input_w + w_in;
                    
                    // Use __ldg for read-only data access
                    const scalar_t val = __ldg(&input[input_idx]);
                    if (val > max_val) {
                        max_val = val;
                        max_index = input_idx;
                    }
                }
            }
        }

        // Aligned store operations
        output[idx] = max_val;
        if (indices != nullptr) {
            indices[idx] = max_index;
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

    // Adjust block size to account for 4-element processing per thread
    const int threads = 256;
    const int total = batch_size * channels * output_d * output_h * output_w;
    const int blocks = (total + (threads * 4) - 1) / (threads * 4);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cuda", ([&] {
        max_pool3d_forward_kernel_aligned<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool3d_cuda_forward, "Max Pool 3D forward aligned (CUDA)");
}