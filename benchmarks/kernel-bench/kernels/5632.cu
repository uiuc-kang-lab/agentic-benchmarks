#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// This kernel uses a grid-stride loop to evenly distribute work across all threads
// Each thread processes multiple output elements if necessary, ensuring load balancing.

template <typename scalar_t>
__global__ void even_workload_maxpool2d_kernel(
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
    
    // Use grid-stride loop for balanced workload distribution
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    
    for (int out_idx = idx; out_idx < total_elements; out_idx += gridSize) {
        // Compute indices for output tensor: b, c, oh, ow
        int ow = out_idx % output_width;
        int oh = (out_idx / output_width) % output_height;
        int c  = (out_idx / (output_width * output_height)) % channels;
        int b  = out_idx / (output_width * output_height * channels);

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        // Iterate over the pooling window
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = oh * stride - padding + kh * dilation;
                int iw = ow * stride - padding + kw * dilation;
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    int input_idx = b * channels * input_height * input_width +
                                    c * input_height * input_width +
                                    ih * input_width +
                                    iw;
                    scalar_t val = input[input_idx];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
        }
        output[out_idx] = max_val;
    }
}

// Host function: prepares kernel launch parameters and dispatches the kernel

torch::Tensor even_workload_maxpool2d_cuda_forward(
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

    // Compute output dimensions
    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Total number of output elements
    const int num_elements = batch_size * channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "even_workload_maxpool2d_cuda_forward", ([&] {
        even_workload_maxpool2d_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &even_workload_maxpool2d_cuda_forward, "Evenly Distributed Max Pool 2D forward (CUDA)");
}
