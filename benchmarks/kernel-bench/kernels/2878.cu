#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename T>
__device__ __forceinline__ T sigmoid_compute(T val) {
    T exp_val = expf(-val);
    return 1.0f / (1.0f + exp_val);
}

template<typename scalar_t>
__global__ void sigmoid_kernel_vectorized(const scalar_t* __restrict__ input,
                                         scalar_t* __restrict__ output,
                                         const int64_t size,
                                         const int64_t chunk_size) {
    constexpr int vec_size = sizeof(float4) / sizeof(scalar_t);
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    // Adjust indices to work within chunk boundaries
    int chunk_start = blockIdx.y * chunk_size;
    int chunk_end = min(chunk_start + chunk_size, size);
    
    for (int i = chunk_start + tid * vec_size; i < chunk_end; i += stride * vec_size) {
        float4 chunk;
        chunk.x = sigmoid_compute(static_cast<float>(input[i]));
        if (i + 1 < chunk_end) chunk.y = sigmoid_compute(static_cast<float>(input[i+1]));
        if (i + 2 < chunk_end) chunk.z = sigmoid_compute(static_cast<float>(input[i+2]));
        if (i + 3 < chunk_end) chunk.w = sigmoid_compute(static_cast<float>(input[i+3]));

        *reinterpret_cast<float4*>(&output[i]) = chunk;
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    
    const int num_streams = 4;
    const int64_t chunk_size = (size + num_streams - 1) / num_streams;
    const int threads = 256;
    const int blocks_per_chunk = (chunk_size + threads * 4 - 1) / (threads * 4);
    
    dim3 grid(blocks_per_chunk, num_streams);
    
    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
        for (int i = 0; i < num_streams; i++) {
            sigmoid_kernel_vectorized<scalar_t><<<grid, threads, 0, streams[i]>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                size,
                chunk_size
            );
        }
    });

    // Synchronize and cleanup streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Streamed Vectorized Sigmoid forward (CUDA)");
}