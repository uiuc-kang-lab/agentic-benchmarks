#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Device function that computes the max pooling result for one output element
// If static_kernel_size > 0, the loops are unrolled and __ldg() is used for cache optimization
// When static_kernel_size == -1, we use a dynamic loop based on the runtime kernel_size

template <typename scalar_t, int static_kernel_size>
__device__ inline scalar_t max_pool2d_compute(
    const scalar_t* __restrict__ input,
    int input_base,
    int input_width,
    int input_height,
    int oh,
    int ow,
    int kernel_size, // used when static_kernel_size == -1
    int stride,
    int padding,
    int dilation) {

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const int base_h = oh * stride - padding;
    const int base_w = ow * stride - padding;

    if (static_kernel_size > 0) {
        #pragma unroll
        for (int kh = 0; kh < static_kernel_size; kh++) {
            int ih = base_h + kh * dilation;
            if (ih < 0 || ih >= input_height) continue;
            #pragma unroll
            for (int kw = 0; kw < static_kernel_size; kw++) {
                int iw = base_w + kw * dilation;
                if (iw < 0 || iw >= input_width) continue;
                scalar_t val = __ldg(&input[input_base + ih * input_width + iw]);
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    } else {
        // Dynamic kernel size branch
        for (int kh = 0; kh < kernel_size; kh++) {
            int ih = base_h + kh * dilation;
            if (ih < 0 || ih >= input_height) continue;
            for (int kw = 0; kw < kernel_size; kw++) {
                int iw = base_w + kw * dilation;
                if (iw < 0 || iw >= input_width) continue;
                scalar_t val = input[input_base + ih * input_width + iw];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }

    return max_val;
}

// Modular kernel: each thread computes one output element using the helper device function

template <typename scalar_t, int static_kernel_size>
__global__ void max_pool2d_modular_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size, // used only when static_kernel_size == -1
    int stride,
    int padding,
    int dilation
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * output_height * output_width;

    // Grid-stride loop for flexibility
    while (index < total) {
        int ow = index % output_width;
        int oh = (index / output_width) % output_height;
        int c  = (index / (output_width * output_height)) % channels;
        int b  = index / (output_width * output_height * channels);
        int input_base = (b * channels + c) * input_height * input_width;

        scalar_t max_val = max_pool2d_compute<scalar_t, static_kernel_size>(
            input,
            input_base,
            input_width,
            input_height,
            oh,
            ow,
            kernel_size, // only used if static_kernel_size == -1
            stride,
            padding,
            dilation
        );
        output[index] = max_val;
        index += blockDim.x * gridDim.x;
    }
}

// Host function that computes output dimensions, allocates output tensor, and launches the kernel

torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    int output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    int total = batch_size * channels * output_height * output_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            max_pool2d_modular_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                channels,
                input_height,
                input_width,
                output_height,
                output_width,
                kernel_size,    // not used in static unroll
                stride,
                padding,
                dilation
            );
        } else if (kernel_size == 3) {
            max_pool2d_modular_kernel<scalar_t, 3><<<blocks, threads>>>(
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
        } else {
            max_pool2d_modular_kernel<scalar_t, -1><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Modular Max Pool 2D forward (CUDA) using device functions");
}
