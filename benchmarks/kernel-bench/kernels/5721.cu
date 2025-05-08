#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Combined kernel that minimizes warp divergence by precomputing valid ranges (kernel2) and
// uses loop unrolling for common kernel sizes (kernel1) to improve performance.

template <typename scalar_t>
__global__ void max_pool2d_combined_kernel(
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
    // Use 1D indexing for all output elements
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * output_height * output_width;
    if (output_idx >= total_outputs) return;

    // Calculate indices for output: ow, oh, channel and batch
    int ow = output_idx % output_width;
    int oh = (output_idx / output_width) % output_height;
    int c  = (output_idx / (output_width * output_height)) % channels;
    int b  = output_idx / (output_width * output_height * channels);

    // Compute the top-left corner in the input corresponding to this output element
    int base_h = oh * stride - padding;
    int base_w = ow * stride - padding;

    // Offset to the beginning of the corresponding input channel
    int input_channel_offset = (b * channels + c) * input_height * input_width;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // For common kernel sizes, use unrolled loops for efficiency
    if (kernel_size == 2) {
        #pragma unroll
        for (int kh = 0; kh < 2; kh++) {
            #pragma unroll
            for (int kw = 0; kw < 2; kw++) {
                int ih = base_h + kh * dilation;
                int iw = base_w + kw * dilation;
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    int input_idx = input_channel_offset + ih * input_width + iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    } else if (kernel_size == 3) {
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            #pragma unroll
            for (int kw = 0; kw < 3; kw++) {
                int ih = base_h + kh * dilation;
                int iw = base_w + kw * dilation;
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    int input_idx = input_channel_offset + ih * input_width + iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    } else {
        // Precompute valid kernel index ranges to minimize divergence
        int kh_start = 0;
        if (base_h < 0) {
            kh_start = (-base_h + dilation - 1) / dilation;
        }
        int kh_end = kernel_size;
        if (base_h + (kernel_size - 1) * dilation >= input_height) {
            int possible_end = (input_height - base_h + dilation - 1) / dilation;
            kh_end = possible_end < kernel_size ? possible_end : kernel_size;
        }

        int kw_start = 0;
        if (base_w < 0) {
            kw_start = (-base_w + dilation - 1) / dilation;
        }
        int kw_end = kernel_size;
        if (base_w + (kernel_size - 1) * dilation >= input_width) {
            int possible_end = (input_width - base_w + dilation - 1) / dilation;
            kw_end = possible_end < kernel_size ? possible_end : kernel_size;
        }

        for (int kh = kh_start; kh < kh_end; ++kh) {
            int ih = base_h + kh * dilation;
            for (int kw = kw_start; kw < kw_end; ++kw) {
                int iw = base_w + kw * dilation;
                int input_idx = input_channel_offset + ih * input_width + iw;
                max_val = max(max_val, input[input_idx]);
            }
        }
    }

    output[output_idx] = max_val;
}


// Forward function that sets up kernel launch

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

    const int num_outputs = batch_size * channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (num_outputs + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_combined_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA) combined kernel");
}
