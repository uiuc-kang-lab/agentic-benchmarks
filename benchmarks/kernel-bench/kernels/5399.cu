#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Kernel that processes a chunk of the batch
template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* input,
    scalar_t* output,
    const int batch_size,  // batch size for this chunk
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * output_height * output_width;
    if (idx >= total) return;

    int ow = idx % output_width;
    int oh = (idx / output_width) % output_height;
    int c  = (idx / (output_width * output_height)) % channels;
    int b  = idx / (channels * output_width * output_height);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int ih = oh * stride - padding + kh * dilation;
            int iw = ow * stride - padding + kw * dilation;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                int in_index = b * (channels * input_height * input_width) +
                               c * (input_height * input_width) +
                               ih * input_width + iw;
                max_val = max(max_val, input[in_index]);
            }
        }
    }
    output[idx] = max_val;
}

// Forward function that pipelines computation via CUDA streams
torch::Tensor max_pool2d_cuda_forward_pipeline(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Ensure input is on CUDA. This non_blocking copy allows overlap if input is on CPU.
    if (!input.is_cuda()) {
        input = input.to(torch::kCUDA, /*non_blocking=*/true);
    }

    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    int output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    // Allocate output on CUDA
    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Use multiple streams to overlap potential memory transfers and kernel execution.
    int num_streams = (batch_size < 4) ? batch_size : 4;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Partition the batch along dimension 0
    int chunk = (batch_size + num_streams - 1) / num_streams;
    const int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward_pipeline", ([&] {
        for (int i = 0; i < num_streams; i++) {
            int start = i * chunk;
            int end = (start + chunk > batch_size) ? batch_size : start + chunk;
            if (start >= end) break;
            int current_batch = end - start;
            int total_elements = current_batch * channels * output_height * output_width;
            int blocks = (total_elements + threads - 1) / threads;

            // Slice the batch for input and output
            const scalar_t* input_ptr = input.data_ptr<scalar_t>() + start * channels * input_height * input_width;
            scalar_t* output_ptr = output.data_ptr<scalar_t>() + start * channels * output_height * output_width;

            max_pool2d_kernel<scalar_t><<<blocks, threads, 0, streams[i]>>>(
                input_ptr,
                output_ptr,
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

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward_pipeline, "Max Pool 2D forward with pipeline (CUDA)");
}
