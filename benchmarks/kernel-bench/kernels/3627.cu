#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_STREAMS 4

template <typename scalar_t>
__global__ void hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   size_t numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        scalar_t x = input[i];
        scalar_t y = (x + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);
        y = y < static_cast<scalar_t>(0) ? static_cast<scalar_t>(0) 
             : (y > static_cast<scalar_t>(1) ? static_cast<scalar_t>(1) : y);
        output[i] = y;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const size_t chunk_size = (numel + NUM_STREAMS - 1) / NUM_STREAMS;
    const int threads = 1024;

    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_cuda", ([&] {
        for (int i = 0; i < NUM_STREAMS; i++) {
            const size_t offset = i * chunk_size;
            const size_t current_chunk_size = std::min(chunk_size, numel - offset);
            if (current_chunk_size > 0) {
                const int blocks = (current_chunk_size + threads - 1) / threads;
                hardsigmoid_kernel<scalar_t><<<blocks, threads, 0, streams[i]>>>(
                    input.data_ptr<scalar_t>() + offset,
                    output.data_ptr<scalar_t>() + offset,
                    current_chunk_size);
            }
        }
    }));

    // Synchronize and destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid activation forward (CUDA)");
}