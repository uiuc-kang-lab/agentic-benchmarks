#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel that minimizes warp divergence using uniform control flow.
__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Compute global thread index and grid stride
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Each thread computes a partial sum using grid-stride loop
    float localSum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        float log_val = log_predictions[i];
        float target = targets[i];
        localSum += expf(log_val) - target * log_val;
    }

    // Perform warp-level reduction using shuffle operations in a uniform manner
    const unsigned int warpSize = 32;
    // Reduce within warp with no divergent branching
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        localSum += __shfl_down_sync(0xffffffff, localSum, offset);
    }
    // Broadcast the final warp sum to all lanes in the warp
    localSum = __shfl_sync(0xffffffff, localSum, 0);

    // Compute lane id using bitwise AND (efficient and branchless)
    int lane = threadIdx.x & (warpSize - 1);
    // Create a leader flag that is 1 for lane 0 and 0 for others (branchless condition)
    int leader = (lane == 0);

    // Only the warp leader contributes its warp's sum via atomic add
    // Multiplying by leader avoids divergent branches
    atomicAdd(output, localSum * leader);
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Launch configuration: use 256 threads per block, ensure blockDim is multiple of warp size
    const int threads = 256;
    const int blocks = min(256, (n + threads - 1) / threads);

    kl_div_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}
