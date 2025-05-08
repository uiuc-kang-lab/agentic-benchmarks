#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel(
    const float4* __restrict__ log_predictions4,
    const float4* __restrict__ targets4,
    float* __restrict__ output,
    const int n4,
    const int n) {
    
    extern __shared__ float shared[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_threads = blockDim.x;
    const int total_blocks = gridDim.x;
    
    // Calculate workload per thread
    const int items_per_thread = (n4 + total_blocks * num_threads - 1) / (total_blocks * num_threads);
    const int start_idx = (bid * num_threads + tid) * items_per_thread;
    const int end_idx = min(start_idx + items_per_thread, n4);
    
    float thread_sum = 0.0f;
    
    // Process 4 elements at a time using float4
    for (int i = start_idx; i < end_idx; i++) {
        float4 log_pred4 = log_predictions4[i];
        float4 target4 = targets4[i];
        
        thread_sum += expf(log_pred4.x) - target4.x * log_pred4.x;
        thread_sum += expf(log_pred4.y) - target4.y * log_pred4.y;
        thread_sum += expf(log_pred4.z) - target4.z * log_pred4.z;
        thread_sum += expf(log_pred4.w) - target4.w * log_pred4.w;
    }
    
    // Handle remaining elements
    const int remaining_start = end_idx * 4;
    const int remaining_end = min(remaining_start + 4, n);
    if (remaining_start < n) {
        const float* log_pred = reinterpret_cast<const float*>(log_predictions4);
        const float* target = reinterpret_cast<const float*>(targets4);
        
        for (int i = remaining_start; i < remaining_end; i++) {
            thread_sum += expf(log_pred[i]) - target[i] * log_pred[i];
        }
    }
    
    // Warp-level reduction
    thread_sum = warp_reduce(thread_sum);
    
    // Store warp results in shared memory
    if (tid % 32 == 0) {
        shared[tid / 32] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < (num_threads / 32)) {
        float warp_sum = shared[tid];
        warp_sum = warp_reduce(warp_sum);
        
        if (tid == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    const int n4 = n / 4;
    
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Dynamic block size based on input size
    const int threads = 256;
    const int min_elements_per_thread = 16;
    const int desired_blocks = (n4 + min_elements_per_thread * threads - 1) / (min_elements_per_thread * threads);
    const int blocks = min(desired_blocks, 1024);
    
    const int warps_per_block = threads / 32;
    const int shared_mem = warps_per_block * sizeof(float);
    
    kl_div_kernel<<<blocks, threads, shared_mem>>>(
        reinterpret_cast<const float4*>(log_predictions.data_ptr<float>()),
        reinterpret_cast<const float4*>(targets.data_ptr<float>()),
        output.data_ptr<float>(),
        n4,
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}