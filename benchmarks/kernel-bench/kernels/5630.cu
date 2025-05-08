#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// CUDA kernel: each kernel call processes a subset (chunk) of the batches
// It is identical to the basic version except that the pointer parameters have been offset
// so that batch indices run from 0 to current_batch_size-1 in the kernel.

template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int current_batch_size,
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
    const int total_elements = current_batch_size * channels * output_height * output_width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int ow = idx % output_width;
    int oh = (idx / output_width) % output_height;
    int c = (idx / (output_width * output_height)) % channels;
    int b = idx / (output_width * output_height * channels);

    // Start with the lowest possible value for maximum
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

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
                max_val = (val > max_val) ? val : max_val;
            }
        }
    }
    output[idx] = max_val;
}

// Host function: splits the input batch dimension into chunks and launches the kernel
// in separate CUDA streams to overlap computation and memory operations.

torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Use multiple streams to partition work across the batch dimension
    int num_streams = (batch_size < 4) ? batch_size : 4;  // Use up to 4 streams
    int chunk = (batch_size + num_streams - 1) / num_streams;  
    const int threads = 256;

    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();

        for (int i = 0; i < num_streams; i++) {
            int start_batch = i * chunk;
            int current_batch = ((start_batch + chunk) > batch_size) ? (batch_size - start_batch) : chunk;
            if (current_batch <= 0) continue;

            int total_out_elements = current_batch * channels * output_height * output_width;
            int blocks = (total_out_elements + threads - 1) / threads;

            // Offset the pointers so that the kernel sees a contiguous batch of size current_batch
            const scalar_t* chunk_input = input_ptr + start_batch * channels * input_height * input_width;
            scalar_t* chunk_output = output_ptr + start_batch * channels * output_height * output_width;

            max_pool2d_kernel<scalar_t><<<blocks, threads, 0, streams[i]>>>(
                chunk_input,
                chunk_output,
                current_batch,
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

    // Synchronize and destroy streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Optimized Max Pool 2D forward with streams (CUDA)");
}
