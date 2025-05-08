#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>

// GELU function specializations

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

// Kernel that applies GELU element-wise using grid-stride loop

template <typename scalar_t>
__global__ void gelu_kernel_overlap(const scalar_t* __restrict__ in,
                                      scalar_t* __restrict__ out,
                                      size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        scalar_t val = in[i];
        out[i] = gelu_function<scalar_t>(val);
    }
}

// Forward function using CUDA streams to overlap memory transfers and computation

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    // Create output tensor
    auto output = torch::empty_like(x);
    size_t total = x.numel();

    // Define chunk size (number of elements per chunk).
    // Adjust CHUNK_SIZE if necessary to match your data and hardware characteristics.
    const size_t CHUNK_SIZE = 1 << 20; // 1M elements per chunk

    // For small tensors, fall back to a single kernel launch without stream overlap
    if (total <= CHUNK_SIZE) {
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda_overlap_single", ([&] {
            gelu_kernel_overlap<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                total);
        }));
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
        return output;
    }

    // Create two CUDA streams for double buffering
    cudaStream_t streams[2];
    cudaError_t err = cudaStreamCreate(&streams[0]);
    TORCH_CHECK(err == cudaSuccess, "Failed to create stream 0: ", cudaGetErrorString(err));
    err = cudaStreamCreate(&streams[1]);
    TORCH_CHECK(err == cudaSuccess, "Failed to create stream 1: ", cudaGetErrorString(err));

    // Allocate double buffers (temporary device memory) for input and output chunks
    void* d_in[2];
    void* d_out[2];
    size_t buffer_size_bytes = CHUNK_SIZE * x.element_size();

    for (int i = 0; i < 2; i++) {
        err = cudaMalloc(&d_in[i], buffer_size_bytes);
        TORCH_CHECK(err == cudaSuccess, "cudaMalloc failed for d_in[" + std::to_string(i) + "]: ", cudaGetErrorString(err));
        err = cudaMalloc(&d_out[i], buffer_size_bytes);
        TORCH_CHECK(err == cudaSuccess, "cudaMalloc failed for d_out[" + std::to_string(i) + "]: ", cudaGetErrorString(err));
    }

    int threads = 256;

    // Process the input tensor in chunks, overlapping async memcpy and kernel execution
    for (size_t offset = 0; offset < total; offset += CHUNK_SIZE) {
        size_t current_chunk = std::min(CHUNK_SIZE, total - offset);
        int buf_idx = (offset / CHUNK_SIZE) % 2;
        cudaStream_t stream = streams[buf_idx];

        // Asynchronously copy current chunk from input tensor to temporary buffer d_in
        err = cudaMemcpyAsync(d_in[buf_idx],
                              reinterpret_cast<const char*>(x.data_ptr()) + offset * x.element_size(),  // pointer arithmetic in bytes is handled by element pointer
                              current_chunk * x.element_size(),
                              cudaMemcpyDeviceToDevice, stream);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync (in) failed: ", cudaGetErrorString(err));

        // Launch the GELU kernel on the current chunk
        int blocks = (current_chunk + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda_overlap_chunk", ([&] {
            gelu_kernel_overlap<scalar_t><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const scalar_t*>(d_in[buf_idx]),
                reinterpret_cast<scalar_t*>(d_out[buf_idx]),
                current_chunk);
        }));

        // Asynchronously copy the result from temporary buffer d_out to the output tensor
        err = cudaMemcpyAsync(output.data_ptr() + offset,
                              d_out[buf_idx],
                              current_chunk * x.element_size(),
                              cudaMemcpyDeviceToDevice, stream);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync (out) failed: ", cudaGetErrorString(err));
    }

    // Synchronize both streams to ensure all operations are complete
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    // Free temporary buffers and destroy streams
    for (int i = 0; i < 2; i++) {
        cudaFree(d_in[i]);
        cudaFree(d_out[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward with overlapped computation and memory transfers (CUDA)");
}
