#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// CUDA kernel with precomputed valid ranges to avoid branch divergence inside loops

template <typename scalar_t>
__global__ void max_pool2d_kernel(
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
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_channel = blockIdx.z;
    if (ow >= output_width || oh >= output_height) return;
    const int c = batch_channel % channels;
    const int b = batch_channel / channels;

    // Compute output tensor coordinates
    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    // Determine the top-left corner of the pooling window in the input
    const int h_start = oh * stride - padding;
    const int w_start = ow * stride - padding;

    // Precompute valid kernel index boundaries for the height dimension
    int kh_start = (h_start < 0) ? ((-h_start + dilation - 1) / dilation) : 0;
    int kh_end = kernel_size;
    if (h_start + (kernel_size - 1) * dilation >= input_height) {
        kh_end = (input_height - h_start + dilation - 1) / dilation;
        if (kh_end < 0) kh_end = 0; // Ensure non-negative range
    }

    // Precompute valid kernel index boundaries for the width dimension
    int kw_start = (w_start < 0) ? ((-w_start + dilation - 1) / dilation) : 0;
    int kw_end = kernel_size;
    if (w_start + (kernel_size - 1) * dilation >= input_width) {
        kw_end = (input_width - w_start + dilation - 1) / dilation;
        if (kw_end < 0) kw_end = 0;
    }

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Loop only over valid indices without runtime branching
    for (int kh = kh_start; kh < kh_end; kh++) {
        const int ih = h_start + kh * dilation;
        for (int kw = kw_start; kw < kw_end; kw++) {
            const int iw = w_start + kw * dilation;
            const int input_idx = b * (channels * input_height * input_width) +
                                  c * (input_height * input_width) +
                                  ih * input_width + iw;
            max_val = max(max_val, input[input_idx]);
        }
    }
    
    output[output_idx] = max_val;
}


// C++ interface for the CUDA kernel
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

    const int threads = 256;
    const int total_outputs = batch_size * channels * output_height * output_width;
    const int blocks = (total_outputs + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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
