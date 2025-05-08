#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Computes the KL divergence for a single element
__device__ inline float compute_kldiv(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

// Processes 4 elements (float4) in a modular way
__device__ inline float process_float4(const float* log_ptr, const float* target_ptr) {
    float4 log_vec = *reinterpret_cast<const float4*>(log_ptr);
    float4 target_vec = *reinterpret_cast<const float4*>(target_ptr);
    return compute_kldiv(log_vec.x, target_vec.x) +
           compute_kldiv(log_vec.y, target_vec.y) +
           compute_kldiv(log_vec.z, target_vec.z) +
           compute_kldiv(log_vec.w, target_vec.w);
}

// Warp-level reduction using shuffle instructions
__device__ inline float warp_reduce(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using modular helper function
__device__ inline float block_reduce(float val, int blockSize) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;  // thread index within warp
    int warpId = threadIdx.x / 32; // warp index
    
    // Each warp performs its own reduction
    val = warp_reduce(val);

    // Write reduced value of each warp to shared memory
    if (lane == 0) {
        shared[warpId] = val;
    }
    __syncthreads();

    // Determine number of warps in the block
    int numWarps = (blockSize + 31) / 32;
    // Let the first warp load all warp sums
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0f;
    
    // Final reduction within first warp
    if (warpId == 0) {
        val = warp_reduce(val);
    }
    return val;
}

// Kernel that uses modular device functions in a grid-stride loop with vectorized accesses
__global__ void modular_kldiv_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n
) {
    int tid = threadIdx.x;
    int globalThreadId = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    
    // Process complete groups of 4 elements
    int vec_n = n / 4;  // number of float4 iterations
    for (int i = globalThreadId; i < vec_n; i += stride) {
        int base = i * 4;
        local_sum += process_float4(log_predictions + base, targets + base);
    }
    
    // Reduce sums within the block
    float sum = block_reduce(local_sum, blockDim.x);
    if (tid == 0) {
        atomicAdd(output, sum);
    }
    
    // Handle tail elements (if n is not divisible by 4) in a modular way
    if (blockIdx.x == 0 && tid == 0) {
        float tail_sum = 0.0f;
        int tail_start = vec_n * 4;
        for (int i = tail_start; i < n; i++) {
            tail_sum += compute_kldiv(log_predictions[i], targets[i]);
        }
        atomicAdd(output, tail_sum);
    }
}

// Host function to launch the modular kernel
torch::Tensor kl_div_cuda_forward_modular(
    torch::Tensor log_predictions,
    torch::Tensor targets
) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    int vec_n = n / 4;  // Number of vectorized iterations
    const int blocks = min((vec_n + threads - 1) / threads, 1024);

    modular_kldiv_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_modular, "Modular KLDiv forward (CUDA)");
}
