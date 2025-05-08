#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float branch_free_hinge_loss(float pred, float target) {
    float prod = pred * target;
    // Use arithmetic operations instead of branching
    // max(0, 1-x) = (1-x + |1-x|) / 2
    float diff = 1.0f - prod;
    float abs_diff = fabsf(diff);
    return (diff + abs_diff) * 0.5f;
}

__global__ void hinge_loss_kernel(const float* __restrict__ predictions, 
                                 const float* __restrict__ targets,
                                 float* __restrict__ output,
                                 const int n) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Grid-stride loop
    while(idx < n) {
        float pred = predictions[idx];
        float target = targets[idx];
        output[idx] = branch_free_hinge_loss(pred, target);
        idx += stride;
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    // Optimize thread and block count for better occupancy
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 65535);

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