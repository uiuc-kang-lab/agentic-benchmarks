#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// This kernel uses a grid-stride loop and employs a tunable block size for optimal performance on the target hardware.
// The block size is provided by the host at runtime (e.g., 32, 64, 128, 256, or 512) to allow for performance experimentation.

template <typename scalar_t>
__global__ void max_pool2d_dynamic_kernel(
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
    const int total = batch_size * channels * output_height * output_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gridSize = blockDim.x * gridDim.x;

    for(; idx < total; idx += gridSize) {
        // Decompose flat index into (b, c, oh, ow)
        int ow = idx % output_width;
        int oh = (idx / output_width) % output_height;
        int c  = (idx / (output_width * output_height)) % channels;
        int b  = idx / (output_width * output_height * channels);

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        int base = b * channels * input_height * input_width + c * input_height * input_width;

        // Iterate over the pooling window
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = oh * stride - padding + kh * dilation;
                int iw = ow * stride - padding + kw * dilation;
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    int input_idx = base + ih * input_width + iw;
                    scalar_t val = __ldg(&input[input_idx]);
                    max_val = (val > max_val) ? val : max_val;
                }
            }
        }
        output[idx] = max_val;
    }
}

// Host function with dynamic block size tuning
// An extra parameter 'block_size' (suggested values: 32, 64, 128, 256, 512) is added to allow experiments with different thread block sizes.

torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int block_size  // new parameter for tunable block size
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int total_elements = batch_size * channels * output_height * output_width;
    int threads = block_size;  // use the block size provided
    int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_dynamic_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, channels,
            input_height, input_width,
            output_height, output_width,
            kernel_size, stride, padding, dilation
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with dynamic block size tuning (CUDA)",
          py::arg("input"), py::arg("kernel_size"), py::arg("stride"), py::arg("padding"),
          py::arg("dilation"), py::arg("block_size") = 256);
}
