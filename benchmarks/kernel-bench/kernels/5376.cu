/*
 * A combined optimized CUDA kernel for 2D max pooling, merging coalesced memory access
 * and loop unrolling for small kernel sizes.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>


// Optimized kernel: uses 2D grid for spatial coalescing and loop unrolling for kernel_size 2 & 3
template <typename scalar_t>
__global__ void max_pool2d_optimized_kernel(
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
    // Determine output spatial position from thread indices
    int ow = threadIdx.x + blockIdx.x * blockDim.x;
    int oh = threadIdx.y + blockIdx.y * blockDim.y;

    // Each block in z dimension represents a (batch, channel) pair
    int bc = blockIdx.z;
    if (ow >= output_width || oh >= output_height) return;

    int b = bc / channels;
    int c = bc % channels;

    // Compute the flat output index (row-major ordering: batch, channel, height, width)
    int out_idx = ((b * channels + c) * output_height + oh) * output_width + ow;

    // Initialize max value to negative infinity
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Determine the starting position in the input tensor
    int h_start = oh * stride - padding;
    int w_start = ow * stride - padding;

    // Use loop unrolling for common small kernel sizes
    if (kernel_size == 2) {
        #pragma unroll
        for (int kh = 0; kh < 2; kh++) {
            int ih = h_start + kh * dilation;
            if (ih < 0 || ih >= input_height) continue;
            #pragma unroll
            for (int kw = 0; kw < 2; kw++) {
                int iw = w_start + kw * dilation;
                if (iw < 0 || iw >= input_width) continue;
                int input_idx = ((b * channels + c) * input_height + ih) * input_width + iw;
                max_val = max(max_val, __ldg(&input[input_idx]));
            }
        }
    } else if (kernel_size == 3) {
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            int ih = h_start + kh * dilation;
            if (ih < 0 || ih >= input_height) continue;
            #pragma unroll
            for (int kw = 0; kw < 3; kw++) {
                int iw = w_start + kw * dilation;
                if (iw < 0 || iw >= input_width) continue;
                int input_idx = ((b * channels + c) * input_height + ih) * input_width + iw;
                max_val = max(max_val, __ldg(&input[input_idx]));
            }
        }
    } else {
        // General case for arbitrary kernel_size
        for (int kh = 0; kh < kernel_size; kh++) {
            int ih = h_start + kh * dilation;
            if (ih < 0 || ih >= input_height) continue;
            for (int kw = 0; kw < kernel_size; kw++) {
                int iw = w_start + kw * dilation;
                if (iw < 0 || iw >= input_width) continue;
                int input_idx = ((b * channels + c) * input_height + ih) * input_width + iw;
                max_val = max(max_val, __ldg(&input[input_idx]));
            }
        }
    }

    output[out_idx] = max_val;
}


// Host function to launch the optimized max pooling kernel
// Uses a 2D block configuration for spatial dimensions and a 3D grid for (batch, channel, spatial dims)

torch::Tensor max_pool2d_optimized_cuda_forward(
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

    // Calculate output dimensions
    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Configure block and grid to enforce coalesced memory access.
    // 32 threads in x (matching the warp size) and 8 in y, similar to kernel1.
    dim3 block(32, 8); // Block size aligned to warp size for better performance
    dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_pool2d_optimized_cuda_forward", ([&] {
        max_pool2d_optimized_kernel<scalar_t><<<grid, block>>>(
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
    m.def("forward", &max_pool2d_optimized_cuda_forward, "Max Pool 2D optimized forward (CUDA)");
}
