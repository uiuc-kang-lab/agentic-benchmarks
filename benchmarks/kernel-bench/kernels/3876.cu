#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define NUM_STREAMS 4

__global__ void softsign_kernel_stream(const float* x, float* out, int num_elements, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + offset < num_elements) {
        float val = x[idx + offset];
        out[idx + offset] = val / (1.0f + fabsf(val));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int elements_per_stream = (num_elements + NUM_STREAMS - 1) / NUM_STREAMS;
    int threads = 256;
    
    // Launch kernels in different streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * elements_per_stream;
        int current_elements = std::min(elements_per_stream, num_elements - offset);
        if (current_elements > 0) {
            int blocks = (current_elements + threads - 1) / threads;
            
            softsign_kernel_stream<<<blocks, threads, 0, streams[i]>>>(
                x.data_ptr<float>(), 
                out.data_ptr<float>(), 
                num_elements,
                offset
            );
        }
    }

    // Synchronize and cleanup streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with stream overlap (CUDA)");
}