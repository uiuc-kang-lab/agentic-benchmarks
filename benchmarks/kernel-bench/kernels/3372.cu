#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Optimized Swish kernel using a grid-stride loop to process workloads larger than the number of threads
__global__ void swish_kernel_strided(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    extern __shared__ float shared_x[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    
    // Process data in tiles
    for (int i = idx; i < n; i += stride) {
        // Load data into shared memory
        shared_x[tid] = (i < n) ? x[i] : 0.0f;
        __syncthreads();
        
        // Process data from shared memory
        if (i < n) {
            float v = shared_x[tid];
            float sig = 1.0f / (1.0f + expf(-v));
            y[i] = v * sig;
        }
        __syncthreads();
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    swish_kernel_strided<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass (CUDA) with grid-stride loop and streams");
}
