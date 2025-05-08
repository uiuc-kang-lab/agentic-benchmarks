#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Combined, optimized MaxPool2D kernel using grid-stride loop and __restrict__ pointers
// to improve workload distribution and memory access efficiency.

template <typename scalar_t>
__global__ void combined_maxpool2d_kernel(
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
    // Total number of output elements
    const int total_elements = batch_size * channels * output_height * output_width;
    
    // Grid-stride loop maximizes thread utilization
    for (int out_idx = blockIdx.x * blockDim.x + threadIdx.x; out_idx < total_elements; out_idx += blockDim.x * gridDim.x) {
        // Compute tensor indices for batch, channel, and spatial dimensions
        const int ow = out_idx % output_width;
        const int oh = (out_idx / output_width) % output_height;
        const int c  = (out_idx / (output_width * output_height)) % channels;
        const int b  = out_idx / (output_width * output_height * channels);

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        // Iterate over the pooling window using branchless max computation
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int ih = oh * stride - padding + kh * dilation;
                const int iw = ow * stride - padding + kw * dilation;

                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    const int input_idx = b * (channels * input_height * input_width) +
                                          c * (input_height * input_width) +
                                          ih * input_width +
                                          iw;
                    const scalar_t val = input[input_idx];
                    max_val = (val > max_val) ? val : max_val;
                }
            }
        }

        output[out_idx] = max_val;
    }
}

// Host function: prepares kernel launch parameters and dispatches the kernel
// It combines error-checking from both implementations.

torch::Tensor combined_maxpool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width  = ((input_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int total_elements = batch_size * channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "combined_maxpool2d_cuda_forward", ([&] {
        combined_maxpool2d_kernel<scalar_t><<<blocks, threads>>>(
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

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &combined_maxpool2d_cuda_forward, "Combined Optimized Max Pool 2D forward (CUDA)");
}
