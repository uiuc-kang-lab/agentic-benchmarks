#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void balanced_hinge_loss_kernel(const float* __restrict__ predictions, const float* __restrict__ targets, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        const float pred = __ldg(&predictions[i]);
        const float target = __ldg(&targets[i]);
        sum += fmaxf(0.0f, 1.0f - pred * target);
    }

    atomicAdd(output, sum);
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::zeros({1}, predictions.options());

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    balanced_hinge_loss_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    // Compute the mean of the output tensor
    auto mean = output / n;
    return mean;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced Workload Hinge Loss Forward");
}