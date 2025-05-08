#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Warp-level reduction for max pooling
template <typename scalar_t>
__device__ scalar_t warp_reduce_max(scalar_t val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Kernel using warp-level primitives for max pooling
template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_warp_optimized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding,
    const int dilation
) {
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    const int input_batch_offset = b * (channels * input_height * input_width);
    const int input_channel_offset = c * (input_height * input_width);
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    #pragma unroll
    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
        const int ih = oh * stride - padding + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            const int ih_offset = ih * input_width;
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                const int iw = ow * stride - padding + kw * dilation;
                if (iw >= 0 && iw < input_width) {
                    const int idx = input_batch_offset + input_channel_offset + ih_offset + iw;
                    max_val = max(max_val, __ldg(&input[idx]));
                }
            }
        }
    }

    // Perform warp-level reduction
    max_val = warp_reduce_max(max_val);

    // Write the result of the first thread in the warp
    if (threadIdx.x % warpSize == 0) {
        output[output_idx] = max_val;
    }
}

// Host function to launch the CUDA kernel
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
    const int total_elements = batch_size * channels * output_height * output_width;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_warp_optimized_kernel<scalar_t, 2><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            batch_size, channels, input_height, input_width,
            output_height, output_width, stride, padding, dilation);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Warp-Optimized Max Pool 2D forward (CUDA)");
}
