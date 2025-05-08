#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define NUM_STREAMS 4

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* output, int chunk_size, int offset) {
    __shared__ float shared_predictions[256];
    __shared__ float shared_targets[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < chunk_size) {
        int global_idx = offset + idx;
        output[global_idx] = fmaxf(0.0f, 1.0f - predictions[global_idx] * targets[global_idx]);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);
    
    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int chunk_size = (n + NUM_STREAMS - 1) / NUM_STREAMS;
    int threads = 256;

    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * chunk_size;
        int current_chunk_size = min(chunk_size, n - offset);
        if (current_chunk_size <= 0) break;

        int blocks = (current_chunk_size + threads - 1) / threads;
        
        hinge_loss_kernel<<<blocks, threads, 0, streams[i]>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            current_chunk_size,
            offset
        );
    }

    // Synchronize and destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    auto mean = torch::mean(output);
    return mean;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward (Streamed)");
}