#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define NUM_STREAMS 4

__global__ void leaky_relu_kernel(const float* x, float* out, float negative_slope, int n) {
    extern __shared__ float shared_mem[];
    
    // Global memory index
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (gid < n) {
        shared_mem[tid] = x[gid];
    }
    __syncthreads();
    
    // Process data in shared memory
    if (gid < n) {
        const float val = shared_mem[tid];
        shared_mem[tid] = val > 0 ? val : val * negative_slope;
    }
    __syncthreads();
    
    // Write back to global memory
    if (gid < n) {
        out[gid] = shared_mem[tid];
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();
    int chunk_size = (n + NUM_STREAMS - 1) / NUM_STREAMS;

    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 1024;

    // Process chunks in parallel streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * chunk_size;
        int current_size = std::min(chunk_size, n - offset);
        if (current_size <= 0) break;

        const int blocks = (current_size + threads - 1) / threads;

        leaky_relu_kernel<<<blocks, threads, 0, streams[i]>>>(
            x.data_ptr<float>() + offset,
            out.data_ptr<float>() + offset,
            negative_slope,
            current_size
        );
    }

    // Cleanup streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward (CUDA)");
}