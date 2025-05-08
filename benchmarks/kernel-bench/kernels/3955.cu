#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_kernel(const float* x, float* out, int chunk_start, int chunk_end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int global_idx = chunk_start + idx;
    if (global_idx < chunk_end) {
        float val = x[global_idx];
out[global_idx] = val / (1.0f + fabsf(val));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    int chunk_size = (num_elements + num_streams - 1) / num_streams;
    int threads = 1024;

    for (int i = 0; i < num_streams; ++i) {
        int chunk_start = i * chunk_size;
        int chunk_end = min(chunk_start + chunk_size, num_elements);
        if (chunk_start >= num_elements) break;
        
        int elements_this_chunk = chunk_end - chunk_start;
        int blocks = (elements_this_chunk + threads - 1) / threads;
        
        softsign_kernel<<<blocks, threads, 0, streams[i]>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), chunk_start, chunk_end
        );
    }

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with streams (CUDA)");
}
