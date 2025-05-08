#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Basic hinge loss kernel operating on a chunk of data
__global__ void hinge_loss_kernel(const float* __restrict__ predictions,
                                   const float* __restrict__ targets,
                                   float* __restrict__ output, 
                                   int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float pred = predictions[idx];
        float targ = targets[idx];
        output[idx] = fmaxf(0.0f, 1.0f - pred * targ);
    }
}

// forward function that overlaps kernel execution with asynchronous memory transfers
// by splitting the workload into chunks processed on different CUDA streams.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    // Allocate device output tensor
    torch::Tensor d_output = torch::empty_like(predictions);

    // Determine chunking parameters
    int num_chunks = 4; // Tunable parameter for pipelining
    int chunk_size = (n + num_chunks - 1) / num_chunks;

    // Allocate pinned host memory for asynchronous copy of results
    float* h_output = nullptr;
    cudaError_t err = cudaHostAlloc((void**)&h_output, n * sizeof(float), cudaHostAllocDefault);
    TORCH_CHECK(err == cudaSuccess, "cudaHostAlloc failed");

    // Create CUDA streams for overlapping computation and memory transfers
    std::vector<cudaStream_t> streams(num_chunks);
    for (int i = 0; i < num_chunks; i++) {
        err = cudaStreamCreate(&streams[i]);
        TORCH_CHECK(err == cudaSuccess, "cudaStreamCreate failed");
    }

    // Launch the hinge loss kernel in each stream and asynchronously copy the results to host
    dim3 threads(256);
    for (int i = 0; i < num_chunks; i++) {
        int start = i * chunk_size;
        int current_chunk = std::min(chunk_size, n - start);
        int blocks = (current_chunk + threads.x - 1) / threads.x;

        hinge_loss_kernel<<<blocks, threads, 0, streams[i]>>>(
            predictions.data_ptr<float>() + start,
            targets.data_ptr<float>() + start,
            d_output.data_ptr<float>() + start,
            current_chunk
        );

        // Asynchronously copy the computed chunk from device to host
        err = cudaMemcpyAsync(h_output + start, d_output.data_ptr<float>() + start,
                              current_chunk * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed");
    }

    // Synchronize all streams to ensure all kernels and memory transfers are complete
    for (int i = 0; i < num_chunks; i++) {
        err = cudaStreamSynchronize(streams[i]);
        TORCH_CHECK(err == cudaSuccess, "cudaStreamSynchronize failed");
        cudaStreamDestroy(streams[i]);
    }

    // Compute the mean on the host
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += static_cast<double>(h_output[i]);
    }
    double mean_val = sum / n;

    cudaFreeHost(h_output);

    return torch::tensor(mean_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stream Overlap Hinge Loss Forward");
}
