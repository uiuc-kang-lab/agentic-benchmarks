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
    const int n4) {
    
    extern __shared__ float sdata[];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid & 31;
    const unsigned int warp_id = tid >> 5;
    
    // Each thread processes consecutive elements for better memory coalescing
    unsigned int idx4 = blockIdx.x * blockDim.x + tid;
    const unsigned int grid_stride = blockDim.x * gridDim.x;
    
    float thread_sum = 0.0f;
    
    // Process 4 elements at a time using vectorized loads
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
    
    // Warp-level reduction first (more efficient than going straight to shared memory)
    thread_sum = warp_reduce(thread_sum);
    
    // Store the warp results in shared memory
    if (lane_id == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction using first warp
    if (warp_id == 0) {
        float warp_sum = 0.0f;
        if (lane_id < (blockDim.x >> 5)) { // number of warps
            warp_sum = sdata[lane_id];
        }
        
        warp_sum = warp_reduce(warp_sum);
        
        if (lane_id == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    const int n4 = n >> 2; // number of float4 elements
    
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Ensure alignment for float4
    TORCH_CHECK((uintptr_t)log_predictions.data_ptr() % 16 == 0, "Input tensor must be 16-byte aligned");
    TORCH_CHECK((uintptr_t)targets.data_ptr() % 16 == 0, "Target tensor must be 16-byte aligned");
    
    const int threads = 256;
    const int blocks = min((n4 + threads - 1) / threads, 1024);
    const int shared_mem = (threads >> 5) * sizeof(float); // space for warp results
    
    kl_div_kernel<<<blocks, threads, shared_mem>>>(
        reinterpret_cast<const float4*>(log_predictions.data_ptr<float>()),
        reinterpret_cast<const float4*>(targets.data_ptr<float>()),
        output.data_ptr<float>(),
        n4
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}