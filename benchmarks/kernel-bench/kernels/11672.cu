#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float compute_kl_element(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kl_div_kernel_tuned(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Shared memory for partial sums (one per warp)
    extern __shared__ float warp_sums[];
    
    // Calculate warp information
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    
    // Calculate global index and stride
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Compute local sum with loop unrolling
    float sum = 0.0f;
    
    // Process elements in chunks of 4
    while (idx + 3 * stride < n) {
        sum += compute_kl_element(log_predictions[idx], targets[idx]);
        sum += compute_kl_element(log_predictions[idx + stride], targets[idx + stride]);
        sum += compute_kl_element(log_predictions[idx + 2 * stride], targets[idx + 2 * stride]);
        sum += compute_kl_element(log_predictions[idx + 3 * stride], targets[idx + 3 * stride]);
        idx += 4 * stride;
    }
    
    // Handle remaining elements
    while (idx < n) {
        sum += compute_kl_element(log_predictions[idx], targets[idx]);
        idx += stride;
    }
    
    // Warp-level reduction
    sum = warp_reduce_sum(sum);
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction across warps (done by first warp)
    if (warp_id == 0 && lane < warps_per_block) {
        float warp_sum = warp_sums[lane];
        warp_sum = warp_reduce_sum(warp_sum);
        
        if (lane == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Use 128 threads per block for potentially better occupancy
    const int threads = 128;
    // Adjust number of blocks based on input size and thread count
    const int blocks = min((n + threads - 1) / threads, 2048);
    const int warps_per_block = threads / 32;
    const int shared_mem = warps_per_block * sizeof(float);
    
    kl_div_kernel_tuned<<<blocks, threads, shared_mem>>>(
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