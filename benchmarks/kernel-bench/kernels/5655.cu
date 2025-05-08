#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Kernel using efficient thread and block indexing for improved performance
// Targets 2D grid for channels and output elements for better load balancing

template <typename scalar_t>
__global__ void efficient_maxpool2d_kernel(
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
    // Compute global thread indices
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z;
    const int b = blockIdx.w;

    if (ow >= output_width || oh >= output_height || c >= channels || b >= batch_size) return;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Iterate over the pooling window
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = oh * stride - padding + kh * dilation;
            int iw = ow * stride - padding + kw * dilation;

            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                int input_idx = b * channels * input_height * input_width +
                                c * input_height * input_width +
                                ih * input_width +
                                iw;
                scalar_t val = input[input_idx];
                max_val = max(max_val, val);
            }
        }
    }
    int output_idx = b * channels * output_height * output_width +
                     c * output_height * output_width +
                     oh * output_width +
                     ow;
    output[output_idx] = max_val;
}

// Host function to launch the kernel
torch::Tensor efficient_maxpool2d_cuda_forward(
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

    // Calculate output dimensions
    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    dim3 threads(16, 16);  // Block size of 16x16 threads
    dim3 blocks((output_width + threads.x - 1) / threads.x,
                (output_height + threads.y - 1) / threads.y,
                channels,
                batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "efficient_maxpool2d_cuda_forward", ([&] {
        efficient_maxpool2d_kernel<scalar_t><<<blocks, threads>>>(
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
            dilation);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &efficient_maxpool2d_cuda_forward, "Efficient Max Pool 2D forward (CUDA)");
}
