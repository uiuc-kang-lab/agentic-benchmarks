#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32

__global__ void hinge_loss_warp_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n
) {
    // Calculate warp-aligned indices
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_offset = warp_id * WARP_SIZE;
    
    // Ensure coalesced memory access within warps
    if (warp_offset + lane_id < n) {
        const int idx = warp_offset + lane_id;
        const float pred = predictions[idx];
        const float target = targets[idx];
        output[idx] = fmaxf(0.0f, 1.0f - pred * target);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    const int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    // Configure grid and block dimensions to align with warps
    const int threads_per_block = 256; // Multiple of WARP_SIZE
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    hinge_loss_warp_kernel<<<blocks, threads_per_block>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp Aligned Hinge Loss Forward");
}