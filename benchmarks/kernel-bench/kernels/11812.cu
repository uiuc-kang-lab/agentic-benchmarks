#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kl_div_kernel_tuned(
    const float* log_predictions,
    const float* targets,
    float* output,
    const int n) {

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    
    // Shared memory for partial sums
    __shared__ float shared_sum[4]; // Reduced shared memory size for 128 threads (4 warps)
    
    float thread_sum = 0.0f;
    
    // Grid-stride loop for coalesced memory access
    for(int idx = gid; idx < n; idx += stride) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        thread_sum += __expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction first
    int warp_id = tid / 32;
    float warp_sum = warp_reduce_sum(thread_sum);
    
    // Only first thread in each warp writes to shared memory
    if (tid % 32 == 0) {
        shared_sum[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp)
    if (warp_id == 0 && tid < 4) {
        float final_sum = shared_sum[tid];
        final_sum = warp_reduce_sum(final_sum);
        
        if (tid == 0) {
            atomicAdd(output, final_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimized block size of 128 threads
    const int threads = 128;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = 4 * sizeof(float); // Reduced shared memory size
    
    kl_div_kernel_tuned<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Block size tuned KL divergence forward (CUDA)");
}