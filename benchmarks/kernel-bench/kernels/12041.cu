#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void hinge_loss_warp_optimized_kernel(const float* __restrict__ predictions, const float* __restrict__ targets, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (idx < n) {
        const float pred = __ldg(&predictions[idx]);
        const float target = __ldg(&targets[idx]);
        sum = fmaxf(0.0f, 1.0f - pred * target);
    }
    
    // Perform warp reduction
    sum = warp_reduce_sum(sum);

    // Write reduced sum to global memory
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(output, sum);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::zeros({1}, predictions.options());

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    hinge_loss_warp_optimized_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Hinge Loss Warp Optimized Forward");
}