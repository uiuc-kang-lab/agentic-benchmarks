#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sigmoid_kernel_syncthreads(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            const int64_t size) {
    extern __shared__ float shared_data[];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Load data to shared memory if within bounds
    if (idx < size) {
        shared_data[tid] = input[idx];
    }
    
    // Synchronize to ensure all data is loaded to shared memory
    __syncthreads();

    // Perform sigmoid only if within bound
    if (idx < size) {
        float val = shared_data[tid];
        shared_data[tid] = 1.0f / (1.0f + expf(-val));
    }

    // Synchronize to ensure all threads have completed computation
    __syncthreads();

    // Write results back to global memory
    if (idx < size) {
        output[idx] = shared_data[tid];
    }
}



torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(float);

    sigmoid_kernel_syncthreads<<<blocks, threads, shared_mem_size>>>( 
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid forward (CUDA) with minimal __syncthreads usage");
}
