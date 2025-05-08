#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel for product reduction over an arbitrary dimension size
__global__ void prod_reduce_kernel(const float* input, float* output, int stride, int num_elements, int reduction_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float product = 1.0f;
        for (int i = 0; i < reduction_size; ++i) {
            product *= input[idx + i * stride];
        }
        output[idx] = product;
    }
}

// This forward function uses multiple CUDA streams to overlap kernel computation with asynchronous
// device-to-host memory transfers. The computation is partitioned into chunks, each processed on its
// own stream. Once a chunk's kernel is launched, its result is copied asynchronously to pinned host memory,
// effectively pipelining the work and reducing the overall runtime.

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    // Obtain output shape by removing the reduction dimension
    auto sizes = x.sizes().vec();
    int reduction_size = sizes[dim];
    TORCH_CHECK(reduction_size == 50, "Reduction dimension size must be 50");
    sizes.erase(sizes.begin() + dim);

    // Allocate an intermediate output tensor on the device
    torch::Tensor d_output = torch::empty(sizes, x.options());
    // Allocate pinned host memory for the final output to enable asynchronous transfers
    torch::Tensor h_output = torch::empty(sizes, x.options().device(torch::kCPU).pinned_memory(true));

    int num_elements = d_output.numel();
    int stride = x.stride(dim);
    const float* d_input = x.data_ptr<float>();
    float* d_out = d_output.data_ptr<float>();
    float* h_out = h_output.data_ptr<float>();

    // Set up multiple streams to overlap computation and memory transfers
    int num_streams = 4;
    int chunk_size = (num_elements + num_streams - 1) / num_streams;
    int threads = 1024;
    std::vector<cudaStream_t> streams(num_streams);

    // Create non-blocking streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    for (int i = 0; i < num_streams; i++) {
        int start = i * chunk_size;
        int current_chunk = std::min(chunk_size, num_elements - start);
        if (current_chunk <= 0) break;
        int blocks = (current_chunk + threads - 1) / threads;
        
        // Launch the reduction kernel on the i-th stream for the current chunk
        prod_reduce_kernel<<<blocks, threads, 0, streams[i]>>>(d_input + start, d_out + start, stride, current_chunk);
        
        // Asynchronously copy the computed results from device to pinned host memory
        cudaMemcpyAsync(h_out + start, d_out + start, current_chunk * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize all streams to ensure all operations are complete
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return h_output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension with stream pipelining (CUDA)");
}
