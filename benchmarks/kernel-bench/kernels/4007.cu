#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that processes a chunk of data from index 'start' (inclusive) to 'end' (exclusive)
__global__ void elu_kernel_chunk(const float* x, float* out, float alpha, int start, int end) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end) {
        float val = x[idx];
        out[idx] = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
    }
}

// Host function: splits the input tensor into chunks and processes each chunk in its own CUDA stream
torch::Tensor elu_cuda_overlap(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    // Number of streams to use for overlapping computation with memory operations
    const int num_streams = 4;
    int chunk_size = (n + num_streams - 1) / num_streams;  // Ceiling division to cover all elements
    const int threads = 256;

    // Create CUDA streams
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch a kernel for each chunk in its corresponding stream
    for (int i = 0; i < num_streams; i++) {
        int start = i * chunk_size;
        if (start >= n) break;
        int end = (start + chunk_size < n) ? (start + chunk_size) : n;
        int current_chunk = end - start;
        int blocks = (current_chunk + threads - 1) / threads;
        elu_kernel_chunk<<<blocks, threads, 0, streams[i]>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, start, end);
    }

    // Synchronize and destroy streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_overlap, "ELU activation with overlapped computation and memory transfers using CUDA streams");
}
