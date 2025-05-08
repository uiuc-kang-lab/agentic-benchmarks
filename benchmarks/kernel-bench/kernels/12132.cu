#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ inline float hinge_loss(float pred, float target) {
    return fmaxf(0.0f, 1.0f - pred * target);
}

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* output, int n) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    // Each thread processes multiple elements with a stride
    for(int idx = gid; idx < n; idx += stride) {
        // Ensure coalesced memory access pattern
        float pred = predictions[idx];
        float target = targets[idx];
        output[idx] = hinge_loss(pred, target);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    int threads = 256;
    int blocks = min(65535, (n + threads - 1) / threads);

    hinge_loss_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    auto mean = torch::mean(output);
    return mean;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward");
}