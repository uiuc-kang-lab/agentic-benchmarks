#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>

// Device function: GELU activation applied elementwise
// Specializations for float and double

template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x);

template <>
__device__ inline float gelu_function<float>(float x) {
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

template <>
__device__ inline double gelu_function<double>(double x) {
    return x * 0.5 * (1.0 + erf(x / 1.4142135623730951));
}

// Kernel to compute GELU over a chunk using a grid-stride loop
template <typename scalar_t>
__global__ void gelu_kernel_compute(const scalar_t* __restrict__ in,
                                      scalar_t* __restrict__ out,
                                      size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        scalar_t val = in[i];
        out[i] = gelu_function<scalar_t>(val);
    }
}

// Forward function with pipelined overlapping of memory transfers and computation
// Uses two separate streams (one for memcpy and one for compute) with double buffering and events

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    auto output = torch::empty_like(x);
    size_t total = x.numel();
    
    // Define chunk size (number of elements per chunk) for pipelining
    const size_t CHUNK_SIZE = 1 << 20; // 1M elements per chunk

    // For small tensors, use a single kernel launch on the default stream
    if (total <= CHUNK_SIZE) {
        const int threads = 256;
        int blocks = (total + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_stream_overlap_simple", ([&] {
            gelu_kernel_compute<scalar_t><<<blocks, threads>>>(x.data_ptr<scalar_t>(),
                                                               output.data_ptr<scalar_t>(),
                                                               total);
        }));
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
        return output;
    }

    // Create two CUDA streams: one for memory copies and one for kernel computation
    cudaStream_t stream_copy, stream_compute;
    cudaError_t err = cudaStreamCreate(&stream_copy);
    TORCH_CHECK(err == cudaSuccess, "Failed to create stream_copy: ", cudaGetErrorString(err));
    err = cudaStreamCreate(&stream_compute);
    TORCH_CHECK(err == cudaSuccess, "Failed to create stream_compute: ", cudaGetErrorString(err));

    // Create events for inter-stream synchronization
    cudaEvent_t event_copy_done, event_compute_done;
    err = cudaEventCreateWithFlags(&event_copy_done, cudaEventDisableTiming);
    TORCH_CHECK(err == cudaSuccess, "Failed to create event_copy_done: ", cudaGetErrorString(err));
    err = cudaEventCreateWithFlags(&event_compute_done, cudaEventDisableTiming);
    TORCH_CHECK(err == cudaSuccess, "Failed to create event_compute_done: ", cudaGetErrorString(err));

    // Allocate double buffers for input and output chunks
    void* d_in[2];
    void* d_out[2];
    size_t buf_bytes = CHUNK_SIZE * x.element_size();
    for (int i = 0; i < 2; i++) {
        err = cudaMalloc(&d_in[i], buf_bytes);
        TORCH_CHECK(err == cudaSuccess, "Failed to allocate d_in[", i, "]: ", cudaGetErrorString(err));
        err = cudaMalloc(&d_out[i], buf_bytes);
        TORCH_CHECK(err == cudaSuccess, "Failed to allocate d_out[", i, "]: ", cudaGetErrorString(err));
    }

    // Get raw pointers for the tensor data
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_stream_overlap_pipelined", ([&] {
        auto x_ptr = x.data_ptr<scalar_t>();
        auto y_ptr = output.data_ptr<scalar_t>();

        const int threads = 256;
        size_t offset = 0;
        int chunk_idx = 0;
        while (offset < total) {
            size_t current_chunk = std::min(CHUNK_SIZE, total - offset);
            int buf_idx = chunk_idx % 2;

            // Asynchronously copy input chunk from x to temporary buffer d_in[buf_idx] using stream_copy
            err = cudaMemcpyAsync(d_in[buf_idx],
                                  x_ptr + offset,
                                  current_chunk * sizeof(scalar_t),
                                  cudaMemcpyDeviceToDevice,
                                  stream_copy);
            TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync (input) failed: ", cudaGetErrorString(err));

            // Record event on stream_copy when input copy is complete
            err = cudaEventRecord(event_copy_done, stream_copy);
            TORCH_CHECK(err == cudaSuccess, "cudaEventRecord (copy) failed: ", cudaGetErrorString(err));

            // In stream_compute, wait for the input copy to complete
            err = cudaStreamWaitEvent(stream_compute, event_copy_done, 0);
            TORCH_CHECK(err == cudaSuccess, "cudaStreamWaitEvent (copy) failed: ", cudaGetErrorString(err));

            // Launch the GELU kernel on the current chunk on stream_compute
            int blocks = (current_chunk + threads - 1) / threads;
            gelu_kernel_compute<scalar_t><<<blocks, threads, 0, stream_compute>>>(
                reinterpret_cast<scalar_t*>(d_in[buf_idx]),
                reinterpret_cast<scalar_t*>(d_out[buf_idx]),
                current_chunk);

            // Record event on stream_compute when computation is done
            err = cudaEventRecord(event_compute_done, stream_compute);
            TORCH_CHECK(err == cudaSuccess, "cudaEventRecord (compute) failed: ", cudaGetErrorString(err));

            // In stream_copy, wait for the computation to finish
            err = cudaStreamWaitEvent(stream_copy, event_compute_done, 0);
            TORCH_CHECK(err == cudaSuccess, "cudaStreamWaitEvent (compute) failed: ", cudaGetErrorString(err));

            // Asynchronously copy the computed result from d_out[buf_idx] back to the output tensor
            err = cudaMemcpyAsync(y_ptr + offset,
                                  d_out[buf_idx],
                                  current_chunk * sizeof(scalar_t),
                                  cudaMemcpyDeviceToDevice,
                                  stream_copy);
            TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync (output) failed: ", cudaGetErrorString(err));

            offset += current_chunk;
            chunk_idx++;
        }
    }));

    // Synchronize to ensure all operations are complete
    cudaStreamSynchronize(stream_copy);
    cudaStreamSynchronize(stream_compute);

    // Clean up temporary resources
    for (int i = 0; i < 2; i++) {
        cudaFree(d_in[i]);
        cudaFree(d_out[i]);
    }
    cudaEventDestroy(event_copy_done);
    cudaEventDestroy(event_compute_done);
    cudaStreamDestroy(stream_copy);
    cudaStreamDestroy(stream_compute);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward with overlapped memory transfers and computation (CUDA)");
}
