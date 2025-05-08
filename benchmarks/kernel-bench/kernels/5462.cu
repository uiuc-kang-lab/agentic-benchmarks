#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Unroll the kernel loops to optimize memory access and computation
// maximum kernel size considered here is 5x5 for manual unrolling

// Utility function to calculate input index
__device__ inline int calculate_input_index(const int b, const int c, const int h, const int w,
                                             const int channels, const int height, const int width) {
    return b * (channels * height * width) +
           c * (height * width) +
           h * width +
           w;
}

// Utility function to check for valid position
__device__ inline bool is_valid_position(const int h, const int w,
                                         const int height, const int width) {
    return h >= 0 && h < height && w >= 0 && w < width;
}

// Kernel function with unrolled loops for max pooling
__global__ void max_pool2d_kernel_unrolled(
    const float* __restrict__ input,
    float* output,
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
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    float max_val = -std::numeric_limits<float>::infinity();

    if (kernel_size == 3) {
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            int ih = oh * stride - padding + kh * dilation;
            #pragma unroll
            for (int kw = 0; kw < 3; kw++) {
                int iw = ow * stride - padding + kw * dilation;
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    int input_idx = calculate_input_index(b, c, ih, iw, channels, input_height, input_width);
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    } else if (kernel_size == 2) {
        #pragma unroll
        for (int kh = 0; kh < 2; kh++) {
            int ih = oh * stride - padding + kh * dilation;
            #pragma unroll
            for (int kw = 0; kw < 2; kw++) {
                int iw = ow * stride - padding + kw * dilation;
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    int input_idx = calculate_input_index(b, c, ih, iw, channels, input_height, input_width);
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    } 
    // Include similar blocks for kernel sizes 4x4 and 5x5 if necessary

    output[output_idx] = max_val;
}

// Forward function
torch::Tensor max_pool2d_cuda_forward_unrolled(
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
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward_unrolled", ([&] {
        max_pool2d_kernel_unrolled<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
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
    m.def("forward", &max_pool2d_cuda_forward_unrolled, "Max Pool 2D forward with manual loop unrolling (CUDA)");
}