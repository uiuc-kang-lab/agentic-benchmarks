#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// This kernel combines grid-stride loops with compile-time unrolling for common kernel sizes (2 and 3).
// For static_kernel_size > 0, the kernel window loops are unrolled and __ldg() is used for faster read-only cache accesses.
// When static_kernel_size is -1, the kernel uses a dynamic loop to support arbitrary kernel sizes.

template <typename scalar_t, int static_kernel_size>
__global__ void max_pool2d_efficient_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,  // runtime kernel size when static_kernel_size == -1
    const int stride,
    const int padding,
    const int dilation
) {
    const int total_elements = batch_size * channels * output_height * output_width;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int gridSize = blockDim.x * gridDim.x;

    while (index < total_elements) {
        // Compute 4D indices from the flat index
        int ow = index % output_width;
        int oh = (index / output_width) % output_height;
        int c  = (index / (output_width * output_height)) % channels;
        int b  = index / (output_width * output_height * channels);

        // Base pointer for current (b, c) feature map; precompute offset for efficiency
        int input_offset = (b * channels + c) * input_height * input_width;
        // Initialize maximum value to negative infinity
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        if (static_kernel_size > 0) {
            // Compile-time constant kernel size: unroll loops and use __ldg() for read-only caching
            #pragma unroll
            for (int kh = 0; kh < static_kernel_size; kh++) {
                int ih = oh * stride - padding + kh * dilation;
                if (ih >= 0 && ih < input_height) {
                    #pragma unroll
                    for (int kw = 0; kw < static_kernel_size; kw++) {
                        int iw = ow * stride - padding + kw * dilation;
                        if (iw >= 0 && iw < input_width) {
                            scalar_t val = __ldg(&input[input_offset + ih * input_width + iw]);
                            if (val > max_val)
                                max_val = val;
                        }
                    }
                }
            }
        } else {
            // Dynamic kernel size branch
            for (int kh = 0; kh < kernel_size; kh++) {
                int ih = oh * stride - padding + kh * dilation;
                if (ih >= 0 && ih < input_height) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int iw = ow * stride - padding + kw * dilation;
                        if (iw >= 0 && iw < input_width) {
                            scalar_t val = input[input_offset + ih * input_width + iw];
                            if (val > max_val)
                                max_val = val;
                        }
                    }
                }
            }
        }

        output[index] = max_val;
        index += gridSize;
    }
}

// Host function: calculates output dimensions and dispatches the CUDA kernel based on the provided kernel_size.
// For common small kernel sizes (2 and 3), we dispatch the compile-time optimized kernel; otherwise we use the dynamic branch.

torch::Tensor max_pool2d_cuda_forward(
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

    const int total_elements = batch_size * channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            max_pool2d_efficient_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                channels,
                input_height,
                input_width,
                output_height,
                output_width,
                kernel_size, // not used in this branch
                stride,
                padding,
                dilation
            );
        } else if (kernel_size == 3) {
            max_pool2d_efficient_kernel<scalar_t, 3><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                channels,
                input_height,
                input_width,
                output_height,
                output_width,
                kernel_size, // not used in this branch
                stride,
                padding,
                dilation
            );
        } else {
            max_pool2d_efficient_kernel<scalar_t, -1><<<blocks, threads>>>(
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

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Efficient Max Pool 2D forward (CUDA)");
}
