#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Basic LeakyReLU kernel, unchanged in its core computation
__global__ void leaky_relu_kernel(const float* __restrict__ x, float* __restrict__ out, float negative_slope, int n) {
    __shared__ float shared_x[1024];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        out[idx] = (val > 0.0f) ? val : val * negative_slope;
    }
}

// Forward function that splits the work into chunks and uses multiple CUDA streams
// to overlap kernel execution with memory operations (if any asynchronous transfers occur
// later in the pipeline). This can help reduce overall runtime on GPUs like the NVIDIA H100.

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    // Create output tensor with the same shape as input
    auto out = torch::empty_like(x);
    const int n = x.numel();

    // Define the number of threads per block
    const int threads = 1024;
    
    // Use two streams to pipeline work across chunks
    const int num_streams = 2;
    // Divide the total work among the streams
    const int chunk_size = (n + num_streams - 1) / num_streams;

    // Create CUDA streams
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch the kernel on each chunk asynchronously
    for (int s = 0; s < num_streams; s++) {
        int start = s * chunk_size;
        if (start >= n) break;
        int current_chunk = ((n - start) < chunk_size) ? (n - start) : chunk_size;
        int blocks = (current_chunk + threads - 1) / threads;

        leaky_relu_kernel<<<blocks, threads, 0, streams[s]>>>(
            x.data_ptr<float>() + start,
            out.data_ptr<float>() + start,
            negative_slope,
            current_chunk
        );
    }

    // Synchronize and destroy streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward (CUDA) with streams");
}
