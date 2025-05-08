#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Unified kernel that supports both compile-time unrolling for common kernel sizes
// (e.g., kernel_size==2) and dynamic looping for arbitrary kernel sizes.
// When UNROLL_KERNEL_SIZE > 0, the loops are unrolled (compile-time constant), otherwise dynamic.

template <typename scalar_t, int UNROLL_KERNEL_SIZE>
__global__ void max_pool2d_kernel_combined(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,   // dynamic kernel size (used if UNROLL_KERNEL_SIZE == 0)
    const int stride,
    const int padding,
    const int dilation
) {
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * channels * output_height * output_width)
        return;

    // Compute 4D indices: (b, c, oh, ow)
    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c  = (output_idx / (output_width * output_height)) % channels;
    const int b  = output_idx / (output_width * output_height * channels);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    if (UNROLL_KERNEL_SIZE > 0) {
        // Compile-time unrolling for common kernel sizes
        #pragma unroll
        for (int kh = 0; kh < UNROLL_KERNEL_SIZE; kh++) {
            #pragma unroll
            for (int kw = 0; kw < UNROLL_KERNEL_SIZE; kw++) {
                const int ih = oh * stride - padding + kh * dilation;
                const int iw = ow * stride - padding + kw * dilation;
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    const int input_idx = b * (channels * input_height * input_width) +
                                            c * (input_height * input_width) +
                                            ih * input_width + iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    } else {
        // Use the dynamic kernel_size provided at runtime
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                const int ih = oh * stride - padding + kh * dilation;
                const int iw = ow * stride - padding + kw * dilation;
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    const int input_idx = b * (channels * input_height * input_width) +
                                            c * (input_height * input_width) +
                                            ih * input_width + iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    }

    output[output_idx] = max_val;
}

// Wrapper function: chooses between a specialized unrolled kernel (for kernel_size==2) 
// or a general kernel (for other sizes) and uses a CUDA stream for asynchronous execution.

torch::Tensor max_pool2d_cuda_forward_combined(
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
    const auto output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int total_outputs = batch_size * channels * output_height * output_width;
    const int blocks = (total_outputs + threads - 1) / threads;

    // Create a CUDA stream to overlap computation with memory operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward_combined", ([&] {
        if (kernel_size == 2) {
            // Launch the kernel with compile-time unrolling for kernel_size 2
            max_pool2d_kernel_combined<scalar_t, 2><<<blocks, threads, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                channels,
                input_height,
                input_width,
                output_height,
                output_width,
                kernel_size,  // passed but not used here because of unrolling
                stride,
                padding,
                dilation
            );
        } else {
            // When no unrolling is available, use dynamic looping with kernel_size passed at runtime
            max_pool2d_kernel_combined<scalar_t, 0><<<blocks, threads, 0, stream>>>(
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
        }
    }));

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward_combined, "Combined Max Pool 2D forward (CUDA)");
}
