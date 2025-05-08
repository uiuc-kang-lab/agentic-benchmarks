#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float hinge_loss_compute(float pred, float target) {
    float prod = pred * target;
    // Use arithmetic operations instead of branches
    float diff = 1.0f - prod;
    return fmaxf(diff, 0.0f);
}

__global__ void hinge_loss_kernel(const float* __restrict__ predictions, 
                                const float* __restrict__ targets,
                                float* __restrict__ output,
                                const int n) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;

    for(int idx = gid; idx < n; idx += stride) {
        float pred = predictions[idx];
        float target = targets[idx];
        output[idx] = hinge_loss_compute(pred, target);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    const int threads = 256;
    const int blocks = min(65535, (n + threads - 1) / threads);

    hinge_loss_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward");
}