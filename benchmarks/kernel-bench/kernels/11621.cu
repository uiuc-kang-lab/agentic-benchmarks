#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ void reduce_block(
    float* shared,
    const unsigned int tid,
    const unsigned int wid,
    const unsigned int lane,
    float* output) {
    
    // First reduction: each warp reduces its portion
    float warp_sum = shared[tid];
    warp_sum = warp_reduce_sum(warp_sum);
    
    // Write reduced warp results to shared memory
    if (lane == 0) {
        shared[wid] = warp_sum;
    }
    __syncthreads();
    
    // Final warp reduces all warp results
    if (wid == 0) {
        // Load warp result if available
        float final_sum = (tid < blockDim.x/32) ? shared[tid] : 0.0f;
        final_sum = warp_reduce_sum(final_sum);
        
        if (lane == 0) {
            atomicAdd(output, final_sum);
        }
    }
}

__global__ void kl_div_kernel(
    const float4* __restrict__ log_predictions4,
    const float4* __restrict__ targets4,
    float* __restrict__ output,
    const int n4,
    const int n) {
    
    extern __shared__ float shared[];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int wid = tid >> 5;
    const unsigned int lane = tid & 31;
    
    float thread_sum = 0.0f;
    
    // Process 4 elements at a time using float4
    int idx4 = blockIdx.x * blockDim.x + tid;
    const int grid_stride = gridDim.x * blockDim.x;
    
    while (idx4 < n4) {
        float4 log_pred = log_predictions4[idx4];
        float4 target = targets4[idx4];
        
        // Process vector elements
        thread_sum += expf(log_pred.x) - target.x * log_pred.x;
        thread_sum += expf(log_pred.y) - target.y * log_pred.y;
        thread_sum += expf(log_pred.z) - target.z * log_pred.z;
        thread_sum += expf(log_pred.w) - target.w * log_pred.w;
        
        idx4 += grid_stride;
    }
    
    // Handle remaining elements
    int idx = idx4 * 4;
    const float* log_predictions = reinterpret_cast<const float*>(log_predictions4);
    const float* targets = reinterpret_cast<const float*>(targets4);
    
    while (idx < n) {
        thread_sum += expf(log_predictions[idx]) - targets[idx] * log_predictions[idx];
        idx++;
    }
    
    // Store in shared memory with padding to avoid bank conflicts
    shared[tid] = thread_sum;
    __syncthreads();
    
    // Perform hierarchical reduction
    reduce_block(shared, tid, wid, lane, output);
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    const int n4 = n >> 2;
    
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize thread/block configuration
    const int threads = 256;
    const int blocks = min((n + threads * 4 - 1) / (threads * 4), 1024);
    
    // Shared memory size with padding to avoid bank conflicts
    const int shared_mem = (threads + 32) * sizeof(float);
    
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