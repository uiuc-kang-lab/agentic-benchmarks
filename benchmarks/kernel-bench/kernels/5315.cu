#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <algorithm>

// Kernel to process a sub-batch (chunk) of the input. The input pointer is assumed to point to memory of shape:
// [sub_batch_size, channels, input_height, input_width] in row-major order.
// The output is of shape: [sub_batch_size, channels, output_height, output_width].

template <typename scalar_t>
__global__ void max_pool2d_kernel_stream(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int sub_batch_size,
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
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = sub_batch_size * channels * output_height * output_width;
    if (output_idx >= total_elements) return;

    // Decode the output index into coordinates
    int ow = output_idx % output_width;
    int oh = (output_idx / output_width) % output_height;
    int c  = (output_idx / (output_width * output_height)) % channels;
    int b  = output_idx / (output_width * output_height * channels);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Iterate over the pooling window
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int ih = oh * stride - padding + kh * dilation;
            int iw = ow * stride - padding + kw * dilation;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                int input_idx = b * (channels * input_height * input_width) +
                                c * (input_height * input_width) +
                                ih * input_width +
                                iw;
                max_val = max(max_val, input[input_idx]);
            }
        }
    }

    output[output_idx] = max_val;
}

// This function overlaps host-device memory transfers with kernel computation by processing the input in chunks
// (along the batch dimension) using double buffering with two CUDA streams. For each chunk, a portion of
// the input is asynchronously copied to the device, processed by the kernel, and then the result is
// asynchronously copied back to host memory. This pipelining hides memory latency and improves overall
// throughput without sacrificing computation precision.

torch::Tensor max_pool2d_cuda_forward_streamed(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Ensure input is contiguous. If input resides on CPU, get a pinned version for faster async transfers.
    input = input.contiguous();
    auto input_cpu = input.device().is_cuda() ? input : input.pin_memory();

    // Input shape: [batch_size, channels, input_height, input_width]
    const auto batch_size   = input_cpu.size(0);
    const auto channels     = input_cpu.size(1);
    const auto input_height = input_cpu.size(2);
    const auto input_width  = input_cpu.size(3);

    // Calculate output spatial dimensions
    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width  = ((input_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    // Create output tensor on CPU with pinned memory for async transfers
    auto output_cpu = torch::empty({batch_size, channels, output_height, output_width}, input_cpu.options()).pin_memory();

    // Choose a chunk size along the batch dimension for overlapping transfers (e.g., 16 batches per chunk)
    int chunk_size = (batch_size >= 16) ? 16 : batch_size;

    // Number of elements per batch for input and output
    int input_batch_elems  = channels * input_height * input_width;
    int output_batch_elems = channels * output_height * output_width;

    AT_DISPATCH_FLOATING_TYPES(input_cpu.scalar_type(), "max_pool2d_cuda_forward_streamed", ([&] {
        using scalar_t = scalar_t;
        // Calculate bytes required for one chunk
        size_t input_chunk_bytes  = static_cast<size_t>(chunk_size) * input_batch_elems  * sizeof(scalar_t);
        size_t output_chunk_bytes = static_cast<size_t>(chunk_size) * output_batch_elems * sizeof(scalar_t);

        // Allocate device buffers for one chunk
        scalar_t* d_input = nullptr;
        scalar_t* d_output = nullptr;
        cudaMalloc(&d_input, input_chunk_bytes);
        cudaMalloc(&d_output, output_chunk_bytes);

        // Create two CUDA streams for double buffering
        cudaStream_t streams[2];
        cudaStreamCreate(&streams[0]);
        cudaStreamCreate(&streams[1]);

        scalar_t* input_ptr  = input_cpu.data_ptr<scalar_t>();
        scalar_t* output_ptr = output_cpu.data_ptr<scalar_t>();

        int num_chunks = (batch_size + chunk_size - 1) / chunk_size;
        for (int chunk = 0; chunk < num_chunks; chunk++) {
            int current_chunk_size = std::min(chunk_size, static_cast<int>(batch_size) - chunk * chunk_size);
            int stream_id = chunk % 2;

            // Asynchronously copy a chunk of input from host to device
            cudaMemcpyAsync(
                d_input,
                input_ptr + static_cast<size_t>(chunk) * chunk_size * input_batch_elems,
                static_cast<size_t>(current_chunk_size) * input_batch_elems * sizeof(scalar_t),
                cudaMemcpyHostToDevice,
                streams[stream_id]
            );

            // Compute total number of output elements for this chunk
            int total_outputs = current_chunk_size * channels * output_height * output_width;
            int threads = 256;
            int blocks = (total_outputs + threads - 1) / threads;

            // Launch the kernel on the current stream for this chunk
            max_pool2d_kernel_stream<scalar_t><<<blocks, threads, 0, streams[stream_id]>>>(
                d_input,
                d_output,
                current_chunk_size,
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

            // Asynchronously copy the computed output chunk back from device to host
            cudaMemcpyAsync(
                output_ptr + static_cast<size_t>(chunk) * chunk_size * output_batch_elems,
                d_output,
                static_cast<size_t>(current_chunk_size) * output_batch_elems * sizeof(scalar_t),
                cudaMemcpyDeviceToHost,
                streams[stream_id]
            );
        }

        // Synchronize both streams to ensure all operations complete
        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);

        // Free device buffers and destroy streams
        cudaFree(d_input);
        cudaFree(d_output);
        cudaStreamDestroy(streams[0]);
        cudaStreamDestroy(streams[1]);
    }));

    return output_cpu;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward_streamed, "Max Pool 2D forward with stream overlapping (CUDA)");
}
