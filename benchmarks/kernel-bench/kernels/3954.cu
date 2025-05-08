#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that processes a chunk of the input tensor starting at a given offset
__global__ void softsign_kernel(const float* x, float* out, int offset, int chunk_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < chunk_size) {
        int idx = offset + tid;
        out[idx] = x[idx] / (1.0f + fabsf(x[idx]));
    }
}

// Forward function splitting the work into chunks processed on separate CUDA streams
// This allows overlapping of kernel execution with memory operations (if any) and pipelining of work
// especially beneficial on newer GPUs like the NVIDIA H100 that support concurrent stream execution.

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int num_elements = x.numel();

    // Define the number of streams for pipelining. Adjust nstreams based on workload if needed.
    const int nstreams = 4;
    // Compute the size of each chunk
    int chunk_size = (num_elements + nstreams - 1) / nstreams;

    // Create CUDA streams
    cudaStream_t streams[nstreams];
    for (int i = 0; i < nstreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 1024;

    // Launch kernels on different streams over different chunks of the tensor
    for (int i = 0; i < nstreams; i++) {
        int offset = i * chunk_size;
        int current_chunk = (offset + chunk_size > num_elements) ? (num_elements - offset) : chunk_size;
        if (current_chunk > 0) {
            int blocks = (current_chunk + threads - 1) / threads;
            softsign_kernel<<<blocks, threads, 0, streams[i]>>>(
                x.data_ptr<float>(), out.data_ptr<float>(), offset, current_chunk
            );
        }
    }

    // Synchronize and destroy streams
    for (int i = 0; i < nstreams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation (CUDA) with pipelining using streams");
}
