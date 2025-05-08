#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define UNROLL_FACTOR 8

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int base_idx = tid * UNROLL_FACTOR;
    
    #pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        int idx = base_idx + i;
        if (idx < n) {
            float pred = predictions[idx];
            float target = targets[idx];
            float pred_target = pred * target;
            output[idx] = (pred_target < 1.0f) ? (1.0f - pred_target) : 0.0f;
        }
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    int threads = 256;
    int elements_per_thread = UNROLL_FACTOR;
    int total_threads_needed = (n + elements_per_thread - 1) / elements_per_thread;
    int blocks = (total_threads_needed + threads - 1) / threads;
    blocks = min(blocks, 65535);

    hinge_loss_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward (Unrolled)");
}