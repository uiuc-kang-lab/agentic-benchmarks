#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Optimized MaxPool2D kernel with block size tuning
// to achieve better performance based on hardware and kernel requirements.

template <typename scalar_t>
__global__ void block_size_tuned_maxpool2d_kernel(
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
    
    // Grid-stride loop for balanced workload distribution
    for (int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
         out_idx < total_elements;
         out_idx += blockDim.x * gridDim.x) {

        // Compute indices for output tensor: b, c, oh, ow
        const int ow = out_idx % output_width;
        const int oh = (out_idx / output_width) % output_height;
        const int c  = (out_idx / (output_width * output_height)) % channels;
        const int b  = out_idx / (output_width * output_height * channels);

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        // Iterate over the pooling window
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                const int ih = oh * stride - padding + kh * dilation;
                const int iw = ow * stride - padding + kw * dilation;

                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    const int input_idx = b * channels * input_height * input_width +
                                          c * input_height * input_width +
                                          ih * input_width +
                                          iw;
                    const scalar_t val = input[input_idx];
                    // Use branchless max comparison
                    max_val = (val > max_val) ? val : max_val;
                }
            }
        }
        output[out_idx] = max_val;
    }
}


// Host function preparing CUDA kernel launch parameters and dispatching the kernel

torch::Tensor block_size_tuned_maxpool2d_cuda_forward(
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
    const int threads = 512;  // Experimented block size
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "block_size_tuned_maxpool2d_cuda_forward", ([&] {
        block_size_tuned_maxpool2d_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &block_size_tuned_maxpool2d_cuda_forward, "Block Size Tuned Max Pool 2D forward (CUDA)");
}
