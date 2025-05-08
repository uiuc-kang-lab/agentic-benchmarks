#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Device function to compute max pooling for a given kernel size
// This function is modular and can be reused for different kernel sizes

template <typename scalar_t>
__device__ scalar_t compute_max_pool(
    const scalar_t* __restrict__ input_channel,
    const int input_height,
    const int input_width,
    const int base_ih,
    const int base_iw,
    const int kernel_size,
    const int dilation
) {
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        int ih = base_ih + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            #pragma unroll
            for (int kw = 0; kw < kernel_size; kw++) {
                int iw = base_iw + kw * dilation;
                if (iw >= 0 && iw < input_width) {
                    max_val = max(max_val, input_channel[ih * input_width + iw]);
                }
            }
        }
    }
    return max_val;
}

// Kernel function that uses the modular device function for max pooling

template <typename scalar_t>
__global__ void max_pool2d_modular_kernel(
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
    const int bc = blockIdx.z;
    const int b = bc / channels;
    const int c = bc % channels;

    if (b >= batch_size || c >= channels || oh >= output_height || ow >= output_width)
        return;

    const scalar_t* input_channel = input + (b * channels + c) * input_height * input_width;

    // Precompute base indices
    const int base_ih = oh * stride - padding;
    const int base_iw = ow * stride - padding;

    // Use the device function to compute max pooling
    scalar_t max_val = compute_max_pool(input_channel, input_height, input_width, base_ih, base_iw, kernel_size, dilation);

    output[(b * channels + c) * output_height * output_width + oh * output_width + ow] = max_val;
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

    const dim3 threads(32, 8);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_modular_kernel<scalar_t><<<blocks, threads>>>(
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
