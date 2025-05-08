#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// The kernel processes a partition of the input vector given by an offset.
__global__ void gelu_kernel(const float* x, float* y, int n, int offset) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = offset + i;
    if (idx < n) {
        float xi = x[idx];
        float x_cubed = xi * xi * xi;
        float inner = xi + coeff * x_cubed;
        inner *= sqrt_2_over_pi;
        float tanh_val = tanhf(inner);
        y[idx] = 0.5f * xi * (1.0f + tanh_val);
    }
}

// This function splits the workload into chunks and uses multiple CUDA streams
// to overlap kernel execution (computation) with any concurrent memory operations
// (such as asynchronous copies if integrated in a larger pipeline). On the H100 GPU,
// this increases concurrency and device utilization while ensuring numerical correctness.

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    int n = x.numel();
    auto y = torch::empty_like(x);

    // Choose a number of streams for overlapping operations
    const int num_streams = 4;
    // Compute chunk size per stream
    int chunk_size = (n + num_streams - 1) / num_streams;

    // Create CUDA streams
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 256;
    // Launch a separate kernel for each chunk in its own stream.
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        int remaining = n - offset;
        if (remaining <= 0) break;
        int elements = remaining < chunk_size ? remaining : chunk_size;
        int blocks = (elements + threads - 1) / threads;
        gelu_kernel<<<blocks, threads, 0, streams[i]>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            n,
            offset
        );
    }

    // Synchronize and destroy streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation with overlapping streams");
}
