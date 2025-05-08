#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

template <typename scalar_t>
__global__ void tuned_maxpool3d_kernel(
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

    // Using optimized block dimensions (16x8)
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int linear_idx = blockIdx.z;

    const int d_out = linear_idx % output_d;
    const int tmp = linear_idx / output_d;
    const int c = tmp % channels;
    const int b = tmp / channels;

    if (w_out >= output_w || h_out >= output_h) return;

    // Compute input window bounds
    const int d_start = d_out * stride - padding;
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;

    // Pre-compute valid ranges for pooling window
    const int k_d_start = max(0, (-d_start + dilation - 1) / dilation);
    const int k_d_end = min(kernel_size, (input_d - d_start + dilation - 1) / dilation);
    const int k_h_start = max(0, (-h_start + dilation - 1) / dilation);
    const int k_h_end = min(kernel_size, (input_h - h_start + dilation - 1) / dilation);
    const int k_w_start = max(0, (-w_start + dilation - 1) / dilation);
    const int k_w_end = min(kernel_size, (input_w - w_start + dilation - 1) / dilation);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_index = -1;

    // Base index for current batch and channel
    const int batch_channel_offset = (b * channels + c) * input_d;

    #pragma unroll 4
    for (int kd = k_d_start; kd < k_d_end; kd++) {
        const int d_in = d_start + kd * dilation;
        const int d_offset = batch_channel_offset + d_in;

        #pragma unroll 4
        for (int kh = k_h_start; kh < k_h_end; kh++) {
            const int h_in = h_start + kh * dilation;
            const int h_offset = d_offset * input_h + h_in;

            #pragma unroll 4
            for (int kw = k_w_start; kw < k_w_end; kw++) {
                const int w_in = w_start + kw * dilation;
                const int input_idx = (h_offset * input_w) + w_in;
                const scalar_t val = __ldg(&input[input_idx]);
                if (val > max_val) {
                    max_val = val;
                    max_index = input_idx;
                }
            }
        }
    }

    // Compute output index
    const int output_idx = (((b * channels + c) * output_d + d_out) * output_h + h_out) * output_w + w_out;
    output[output_idx] = max_val;
    if (indices != nullptr) {
        indices[output_idx] = max_index;
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

    // Optimized block dimensions: 16x8 threads (128 threads per block)
    dim3 threads(16, 8);
    dim3 blocks(
        (output_w + threads.x - 1) / threads.x,
        (output_h + threads.y - 1) / threads.y,
        batch_size * channels * output_d
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cuda", ([&] {
        tuned_maxpool3d_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool3d_cuda_forward, "Tuned Max Pool 3D forward (CUDA)");
}