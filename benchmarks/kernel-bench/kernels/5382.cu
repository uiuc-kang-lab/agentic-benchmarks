/*
 * Optimized max pooling CUDA extension using streaming to overlap memory transfers with kernel computation.
 * This implementation splits the batch dimension into chunks processed concurrently on separate CUDA streams.
 * It uses cudaMallocAsync and cudaMemcpyAsync to pipeline host-device data transfers with kernel execution.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <algorithm>


// Optimized kernel: similar to our previous kernel, but processes a chunk (with local batch size).
template <typename scalar_t>
__global__ void max_pool2d_optimized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,  // local batch size for this chunk
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
    // Each thread computes one output element over the chunk
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * output_height * output_width;
    if (output_idx >= total) return;

    int ow = output_idx % output_width;
    int oh = (output_idx / output_width) % output_height;
    int c = (output_idx / (output_width * output_height)) % channels;
    int b = output_idx / (output_width * output_height * channels);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    int h_start = oh * stride - padding;
    int w_start = ow * stride - padding;

    if (kernel_size == 2) {
        #pragma unroll
        for (int kh = 0; kh < 2; kh++) {
            int ih = h_start + kh * dilation;
            if (ih < 0 || ih >= input_height) continue;
            #pragma unroll
            for (int kw = 0; kw < 2; kw++) {
                int iw = w_start + kw * dilation;
                if (iw < 0 || iw >= input_width) continue;
                int input_idx = ((b * channels + c) * input_height + ih) * input_width + iw;
                max_val = max(max_val, __ldg(&input[input_idx]));
            }
        }
    } else if (kernel_size == 3) {
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            int ih = h_start + kh * dilation;
            if (ih < 0 || ih >= input_height) continue;
            #pragma unroll
            for (int kw = 0; kw < 3; kw++) {
                int iw = w_start + kw * dilation;
                if (iw < 0 || iw >= input_width) continue;
                int input_idx = ((b * channels + c) * input_height + ih) * input_width + iw;
                max_val = max(max_val, __ldg(&input[input_idx]));
            }
        }
    } else {
        for (int kh = 0; kh < kernel_size; kh++) {
            int ih = h_start + kh * dilation;
            if (ih < 0 || ih >= input_height) continue;
            for (int kw = 0; kw < kernel_size; kw++) {
                int iw = w_start + kw * dilation;
                if (iw < 0 || iw >= input_width) continue;
                int input_idx = ((b * channels + c) * input_height + ih) * input_width + iw;
                max_val = max(max_val, __ldg(&input[input_idx]));
            }
        }
    }

    output[output_idx] = max_val;
}


/*
 * Host function that uses CUDA streams to overlap memory transfers (from pinned host memory to device)
 * with kernel execution and asynchronous copy of results back to host memory.
 * This implementation splits the batch dimension into chunks and uses cudaMallocAsync, cudaMemcpyAsync,
 * and cudaFreeAsync to pipeline the operations.
 */

torch::Tensor max_pool2d_streaming_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // For this streaming implementation, we assume input resides on CPU. Ensure it is pinned.
    if (!input.is_pinned()) {
        input = input.pin_memory();
    }

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    
    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    // Allocate output tensor on CPU with pinned memory
    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());
    output = output.pin_memory();

    // Decide number of streams to use; here we use 2 streams for pipelining.
    int num_streams = 2;
    // Determine chunk size along the batch dimension
    int chunk_size = (batch_size + num_streams - 1) / num_streams;  // ceiling division

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch operations on each stream. Use AT_DISPATCH to handle different floating types.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_pool2d_streaming_cuda_forward", ([&] {
        using scalar_t_cp = scalar_t;
        for (int i = 0; i < num_streams; i++) {
            int start_batch = i * chunk_size;
            if (start_batch >= batch_size) break;
            int cur_batch = std::min(chunk_size, batch_size - start_batch);

            // Compute number of elements and bytes for input and output chunks
            size_t in_chunk_numel = static_cast<size_t>(cur_batch) * channels * input_height * input_width;
            size_t out_chunk_numel = static_cast<size_t>(cur_batch) * channels * output_height * output_width;
            size_t in_bytes = in_chunk_numel * sizeof(scalar_t_cp);
            size_t out_bytes = out_chunk_numel * sizeof(scalar_t_cp);

            // Allocate device memory for this chunk asynchronously
            scalar_t_cp* d_input = nullptr;
            scalar_t_cp* d_output = nullptr;
            cudaMallocAsync((void**)&d_input, in_bytes, streams[i]);
            cudaMallocAsync((void**)&d_output, out_bytes, streams[i]);

            // Asynchronously copy input chunk from pinned CPU memory to device
            const scalar_t_cp* input_ptr = input.data_ptr<scalar_t_cp>() + start_batch * channels * input_height * input_width;
            cudaMemcpyAsync(d_input, input_ptr, in_bytes, cudaMemcpyHostToDevice, streams[i]);

            // Launch kernel for this chunk
            int total_outputs = cur_batch * channels * output_height * output_width;
            int threads = 256;
            int blocks = (total_outputs + threads - 1) / threads;
            max_pool2d_optimized_kernel<scalar_t_cp><<<blocks, threads, 0, streams[i]>>>(
                d_input,
                d_output,
                cur_batch,
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

            // Asynchronously copy the output chunk back from device to pinned CPU memory
            scalar_t_cp* output_ptr = output.data_ptr<scalar_t_cp>() + start_batch * channels * output_height * output_width;
            cudaMemcpyAsync(output_ptr, d_output, out_bytes, cudaMemcpyDeviceToHost, streams[i]);

            // Free device memory asynchronously
            cudaFreeAsync(d_input, streams[i]);
            cudaFreeAsync(d_output, streams[i]);
        }
    }));

    // Synchronize and destroy streams
    for (auto s : streams) {
        cudaStreamSynchronize(s);
        cudaStreamDestroy(s);
    }

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_streaming_cuda_forward, "Max Pool 2D streaming forward (CUDA) with overlapping transfers");
}
