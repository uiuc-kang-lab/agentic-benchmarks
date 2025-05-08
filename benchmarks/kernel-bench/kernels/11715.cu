#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    
    // Each thread accumulates its own sum
    float thread_sum = 0.0f;
    
    // Step through data with grid stride loop
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_stride = gridDim.x * blockDim.x;
    
    #pragma unroll 4
    while (idx < n) {
        const float log_pred = log_predictions[idx];
        const float target = targets[idx];
        thread_sum += expf(log_pred) - target * log_pred;
        idx += grid_stride;
    }
    
    // Warp-level reduction
    float warp_sum = warpReduceSum(thread_sum);
    
    // Only the first thread in each warp performs atomic add
    if (lane == 0) {
        atomicAdd(output, warp_sum);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Configure kernel launch parameters
    const int blocks = std::min(65535, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    kl_div_kernel<<<blocks, BLOCK_SIZE>>>(
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