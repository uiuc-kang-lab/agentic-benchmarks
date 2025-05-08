#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <cstdint>

// This kernel is specialized for float and supports kernel_size = 2 or 3 with dilation = 1 only
// It uses vectorized 128-bit loads via __ldg() when the input data is 16-byte aligned

template <int KERNEL_SIZE>
__global__ void vectorized_aligned_max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding,
    const int dilation  // dilation must be 1 for vectorized loads
) {
    const int output_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total_elements = batch_size * channels * output_height * output_width;
    if (output_idx >= total_elements) return;

    // Compute output coordinates
    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c  = (output_idx / (output_width * output_height)) % channels;
    const int b  = output_idx / (output_width * output_height * channels);

    float max_val = -INFINITY;
    const int input_batch_stride = channels * input_height * input_width;
    const int input_channel_stride = input_height * input_width;
    const int base = b * input_batch_stride + c * input_channel_stride;
    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;

    // Loop over pooling window rows
    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
        int ih = ih_start + kh * dilation;
        if (ih < 0 || ih >= input_height) continue;
        const float* row_ptr = input + base + ih * input_width;

        // Use vectorized load if possible: only if dilation==1, pooling window is fully within the row,
        // and the starting address is 16-byte aligned
        if (dilation == 1 && iw_start >= 0 && (iw_start + KERNEL_SIZE) <= input_width &&
            (((uintptr_t)(row_ptr + iw_start)) & 0xF) == 0) {
            // Load 128 bits (4 floats) using __ldg
            const float4 vec = __ldg(reinterpret_cast<const float4*>(row_ptr + iw_start));
            if (KERNEL_SIZE == 2) {
                max_val = max(max_val, max(vec.x, vec.y));
            } else if (KERNEL_SIZE == 3) {
                max_val = max(max(max_val, max(vec.x, vec.y)), vec.z);
            }
        } else {
            // Fallback: scalar loads
            for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                int iw = iw_start + kw * dilation;
                if (iw < 0 || iw >= input_width) continue;
                max_val = max(max_val, __ldg(&row_ptr[iw]));
            }
        }
    }

    output[output_idx] = max_val;
}

// Host launcher

torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // This implementation only supports float data type, kernel_size of 2 or 3, and dilation == 1
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Float, "vectorized_aligned_max_pool2d_kernel supports only float type.");
    TORCH_CHECK(kernel_size == 2 || kernel_size == 3, "kernel_size must be 2 or 3 for vectorized_aligned_max_pool2d_kernel.");
    TORCH_CHECK(dilation == 1, "dilation must be 1 for vectorized_aligned_max_pool2d_kernel to use 128-bit aligned vectorized loads.");

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = ((input_height + 2 * padding - kernel_size) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - kernel_size) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());
    const int total_elements = batch_size * channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_vectorized_aligned_forward", ([&] {
        if (kernel_size == 2) {
            vectorized_aligned_max_pool2d_kernel<2><<<blocks, threads>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else if (kernel_size == 3) {
            vectorized_aligned_max_pool2d_kernel<3><<<blocks, threads>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with vectorized aligned memory (CUDA)");
}
