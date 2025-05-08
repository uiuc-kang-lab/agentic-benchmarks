#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 32/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void efficient_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int gid = blockIdx.x * blockDim.x + warp_id * 32 + lane_id;
    const int stride = gridDim.x * blockDim.x;
    
    // Shared memory for warp-level partial sums
    extern __shared__ float shared_mem[];
    shared_mem[tid] = 0.0f;

    float thread_sum = 0.0f;
    // Process elements with grid-level stride
    for (int i = gid; i < n; i += stride) {
        if (i < n) {
            thread_sum += compute_kl_div(log_predictions[i], targets[i]);
        }
    }
    
    // Warp-level reduction
    thread_sum = warp_reduce_sum(thread_sum);
    
    // Store the reduced sum of this warp to shared memory
    if (lane_id == 0) {
        shared_mem[warp_id] = thread_sum;
    }
    __syncthreads();

    // Only first warp's first thread does the final reduction
    if (warp_id == 0 && lane_id == 0) {
        float block_sum = 0.0f;
        for (int i = 0; i < blockDim.x/32; i++) {
            block_sum += shared_mem[i];
        }
        atomicAdd(output, block_sum);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256; // Aligned with a warp multiple for coalescing
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = (threads / 32) * sizeof(float);

    efficient_kl_div_kernel<<<blocks, threads, shared_mem>>>(
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