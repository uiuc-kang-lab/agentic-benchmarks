#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <limits>

// Templated kernel: specialized for kernel_size=2, 3, and generic when KERNEL_SIZE == -1.
// The extra parameter runtime_kernel_size is used in the generic case.

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_tuned_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int stride,
    int padding,
    int dilation,
    int runtime_kernel_size
) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    int ow = output_idx % output_width;
    int oh = (output_idx / output_width) % output_height;
    int c  = (output_idx / (output_width * output_height)) % channels;
    int b  = output_idx / (output_width * output_height * channels);
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    int input_batch_offset = b * (channels * input_height * input_width);
    int input_channel_offset = c * (input_height * input_width);

    if constexpr (KERNEL_SIZE == 2) {
        int ih_base = oh * stride - padding;
        int iw_base = ow * stride - padding;
        if (ih_base >= 0 && ih_base < input_height && iw_base >= 0 && iw_base < input_width) {
            int idx = input_batch_offset + input_channel_offset + ih_base * input_width + iw_base;
            max_val = __ldg(&input[idx]);
        }
        if (ih_base >= 0 && ih_base < input_height && (iw_base + dilation) >= 0 && (iw_base + dilation) < input_width) {
            int idx = input_batch_offset + input_channel_offset + ih_base * input_width + (iw_base + dilation);
            max_val = max(max_val, __ldg(&input[idx]));
        }
        if ((ih_base + dilation) >= 0 && (ih_base + dilation) < input_height && iw_base >= 0 && iw_base < input_width) {
            int idx = input_batch_offset + input_channel_offset + (ih_base + dilation) * input_width + iw_base;
            max_val = max(max_val, __ldg(&input[idx]));
        }
        if ((ih_base + dilation) >= 0 && (ih_base + dilation) < input_height && (iw_base + dilation) >= 0 && (iw_base + dilation) < input_width) {
            int idx = input_batch_offset + input_channel_offset + (ih_base + dilation) * input_width + (iw_base + dilation);
            max_val = max(max_val, __ldg(&input[idx]));
        }
    } else if constexpr (KERNEL_SIZE == 3) {
        int ih_base = oh * stride - padding;
        int iw_base = ow * stride - padding;
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            int ih = ih_base + i * dilation;
            if (ih >= 0 && ih < input_height) {
                int ih_offset = ih * input_width;
                #pragma unroll
                for (int j = 0; j < 3; j++) {
                    int iw = iw_base + j * dilation;
                    if (iw >= 0 && iw < input_width) {
                        int idx = input_batch_offset + input_channel_offset + ih_offset + iw;
                        max_val = max(max_val, __ldg(&input[idx]));
                    }
                }
            }
        }
    } else { // Generic case using runtime_kernel_size
        for (int kh = 0; kh < runtime_kernel_size; kh++) {
            int ih = oh * stride - padding + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                int ih_offset = ih * input_width;
                for (int kw = 0; kw < runtime_kernel_size; kw++) {
                    int iw = ow * stride - padding + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        int idx = input_batch_offset + input_channel_offset + ih_offset + iw;
                        max_val = max(max_val, __ldg(&input[idx]));
                    }
                }
            }
        }
    }

    output[output_idx] = max_val;
}

// The pipelined host function that overlaps host-to-device memory transfers with kernel execution using CUDA streams.
// For CPU input, we split the input tensor along the batch dimension into smaller chunks and use double buffering.
// If the input is already on GPU, we fall back to a single kernel launch without pipelining.

torch::Tensor max_pool2d_cuda_forward_pipelined(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int chunk_size = 32
) {
    // If input is already on GPU, use the non-pipelined path.
    if (input.is_cuda()) {
        const auto batch_size = input.size(0);
        const auto channels = input.size(1);
        const auto input_height = input.size(2);
        const auto input_width = input.size(3);
        int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
        int output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
        auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());
        int total_elements = batch_size * channels * output_height * output_width;
        int threads = 128;
        int blocks = (total_elements + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward_full", ([&] {
            if (kernel_size == 2) {
                max_pool2d_tuned_kernel<scalar_t, 2><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation, 2);
            } else if (kernel_size == 3) {
                max_pool2d_tuned_kernel<scalar_t, 3><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation, 3);
            } else {
                max_pool2d_tuned_kernel<scalar_t, -1><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation, kernel_size);
            }
        }));
        return output;
    }

    // For CPU input, use pipelining with asynchronous memory transfers.
    auto input_pinned = input.pin_memory();
    const auto batch_size = input_pinned.size(0);
    const auto channels = input_pinned.size(1);
    const auto input_height = input_pinned.size(2);
    const auto input_width = input_pinned.size(3);
    int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    int output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    // Allocate output tensor on GPU.
    auto options = input.options().device(torch::kCUDA);
    torch::Tensor output = torch::empty({batch_size, channels, output_height, output_width}, options);

    // Determine maximum chunk sizes in bytes (assuming float type for simplicity).
    size_t in_chunk_max_bytes = chunk_size * channels * input_height * input_width * sizeof(float);
    size_t out_chunk_max_bytes = chunk_size * channels * output_height * output_width * sizeof(float);

    // Create two CUDA streams for double buffering.
    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    // Allocate device buffers for input and output chunks.
    float *d_input0, *d_input1;
    float *d_output0, *d_output1;
    cudaMalloc(&d_input0, in_chunk_max_bytes);
    cudaMalloc(&d_input1, in_chunk_max_bytes);
    cudaMalloc(&d_output0, out_chunk_max_bytes);
    cudaMalloc(&d_output1, out_chunk_max_bytes);

    int n_chunks = (batch_size + chunk_size - 1) / chunk_size;
    for (int i = 0; i < n_chunks; i++) {
        int current_chunk = std::min(chunk_size, static_cast<int>(batch_size - i * chunk_size));
        // Calculate pointers to the current chunk in the pinned input and output tensors.
        float* src_input_ptr = input_pinned.data_ptr<float>() + i * chunk_size * channels * input_height * input_width;
        float* dst_output_ptr = output.data_ptr<float>() + i * chunk_size * channels * output_height * output_width;

        cudaStream_t stream = streams[i % 2];
        float* d_in = (i % 2 == 0) ? d_input0 : d_input1;
        float* d_out = (i % 2 == 0) ? d_output0 : d_output1;

        size_t in_bytes = current_chunk * channels * input_height * input_width * sizeof(float);
        size_t out_bytes = current_chunk * channels * output_height * output_width * sizeof(float);

        // Asynchronously copy the input chunk from CPU to GPU.
        cudaMemcpyAsync(d_in, src_input_ptr, in_bytes, cudaMemcpyHostToDevice, stream);

        // Launch the kernel for this chunk.
        int total_elements = current_chunk * channels * output_height * output_width;
        int threads = 128;
        int blocks = (total_elements + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward_pipelined", ([&] {
            if (kernel_size == 2) {
                max_pool2d_tuned_kernel<scalar_t, 2><<<blocks, threads, 0, stream>>>(
                    d_in, d_out,
                    current_chunk, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation, 2);
            } else if (kernel_size == 3) {
                max_pool2d_tuned_kernel<scalar_t, 3><<<blocks, threads, 0, stream>>>(
                    d_in, d_out,
                    current_chunk, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation, 3);
            } else {
                max_pool2d_tuned_kernel<scalar_t, -1><<<blocks, threads, 0, stream>>>(
                    d_in, d_out,
                    current_chunk, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation, kernel_size);
            }
        }));

        // Asynchronously copy the computed output chunk from the device buffer to the final output tensor on GPU.
        cudaMemcpyAsync(dst_output_ptr, d_out, out_bytes, cudaMemcpyDeviceToDevice, stream);
    }

    // Synchronize the streams to ensure all operations are complete.
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    // Free allocated device buffers and destroy the CUDA streams.
    cudaFree(d_input0);
    cudaFree(d_input1);
    cudaFree(d_output0);
    cudaFree(d_output1);
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward_pipelined, "Max Pool 2D forward pipelined with overlapped memory transfers (CUDA)");
}
