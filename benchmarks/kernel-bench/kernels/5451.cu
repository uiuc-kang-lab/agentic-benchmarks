#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int d_params[4];  // kernel_size, stride, padding, dilation

// Shared memory usage for better memory coalescing
// Assuming maximum kernel size, adjust according to actual maximum
#define MAX_KERNEL_SIZE 11

// Kernel with shared memory optimization
template <typename scalar_t>
__global__ void max_pool2d_kernel_shared(
    const scalar_t* __restrict__ input,
    scalar_t* output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    extern __shared__ scalar_t shared_input[];

    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Load input data into shared memory
    const int shared_offset = (threadIdx.x % d_params[0]) * d_params[0];
    for (int kh = 0; kh < d_params[0]; kh++) {
        #pragma unroll
        for (int kw = 0; kw < d_params[0]; kw++) {
            const int ih = oh * d_params[1] - d_params[2] + kh * d_params[3];
            const int iw = ow * d_params[1] - d_params[2] + kw * d_params[3];
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                const int input_idx = b * (channels * input_height * input_width) +
                                    c * (input_height * input_width) +
                                    ih * input_width +
                                    iw;
                shared_input[shared_offset + kw] = input[input_idx];
            } else {
                shared_input[shared_offset + kw] = -std::numeric_limits<scalar_t>::infinity();
            }
        }
    }
    __syncthreads();

    // Perform max pooling using shared memory
    #pragma unroll
    for (int kh = 0; kh < d_params[0]; kh++) {
        #pragma unroll
        for (int kw = 0; kw < d_params[0]; kw++) {
            max_val = max(max_val, shared_input[shared_offset + kw]);
        }
    }

    output[output_idx] = max_val;
}

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

    // Copy parameters to constant memory
    int h_params[4] = {kernel_size, stride, padding, dilation};
    cudaMemcpyToSymbol(d_params, h_params, sizeof(int) * 4);

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;

    const size_t shared_memory_size = d_params[0] * d_params[0] * sizeof(scalar_t);  // Dynamic shared memory allocation

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel_shared<scalar_t><<<blocks, threads, shared_memory_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}