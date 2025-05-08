#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cstdint>

// Helper function for loading with __ldg and vectorized load if 128-bit aligned

template <typename scalar_t>
__device__ inline scalar_t load_element(const scalar_t* __restrict__ input, int index) {
    return __ldg(input + index);
}

// Specialization for float using vectorized load via float4
template <>
__device__ inline float load_element<float>(const float* __restrict__ input, int index) {
    // Check if the address is 16-byte aligned and index is a multiple of 4
    if ((((uintptr_t)(input + index)) & 0xF) == 0 && ((index & 3) == 0)) {
        const float4* vec_ptr = reinterpret_cast<const float4*>(input + index);
        float4 vec = __ldg(vec_ptr);
        return vec.x;  // Return the first element in the vectorized load
    } else {
        return __ldg(input + index);
    }
}

// Specialization for double using vectorized load via double2 (128-bit load)
template <>
__device__ inline double load_element<double>(const double* __restrict__ input, int index) {
    // Check if the address is 16-byte aligned and index is even
    if ((((uintptr_t)(input + index)) & 0xF) == 0 && ((index & 1) == 0)) {
        const double2* vec_ptr = reinterpret_cast<const double2*>(input + index);
        double2 vec = __ldg(vec_ptr);
        return vec.x;  // Return the first element from the vectorized load
    } else {
        return __ldg(input + index);
    }
}

// CUDA kernel for max pooling with optimized global memory access for read-only data
template <typename scalar_t>
__global__ void max_pool2d_kernel_vectorized_aligned(
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
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    int ow = output_idx % output_width;
    int oh = (output_idx / output_width) % output_height;
    int c = (output_idx / (output_width * output_height)) % channels;
    int b = output_idx / (output_width * output_height * channels);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    int input_channel_stride = input_height * input_width;
    int input_batch_stride = channels * input_channel_stride;

    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        int ih = oh * stride - padding + kh * dilation;
        if (ih < 0 || ih >= input_height) continue;
        int ih_offset = ih * input_width;

        #pragma unroll
        for (int kw = 0; kw < kernel_size; kw++) {
            int iw = ow * stride - padding + kw * dilation;
            if (iw < 0 || iw >= input_width) continue;
            int input_idx = b * input_batch_stride + c * input_channel_stride + ih_offset + iw;
            scalar_t val = load_element<scalar_t>(input, input_idx);
            max_val = (val > max_val ? val : max_val);
        }
    }

    output[output_idx] = max_val;
}

// CUDA forward function
torch::Tensor max_pool2d_cuda_forward_vectorized_aligned(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    int total_output_elements = batch_size * channels * output_height * output_width;
    const int threads = 256;
    int blocks = (total_output_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward_vectorized_aligned", ([&] {
        max_pool2d_kernel_vectorized_aligned<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward_vectorized_aligned, "Max Pool 2D forward with __ldg and 128-bit aligned accesses (CUDA)");
}
