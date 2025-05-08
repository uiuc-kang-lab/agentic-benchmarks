#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel_reduced_atomics(
    const float4* __restrict__ log_predictions_vec,
    const float4* __restrict__ targets_vec,
    float* __restrict__ block_results,
    const int n_vec) {
    
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int global_thread_id = blockIdx.x * blockDim.x + tid;
    
    extern __shared__ float shared_mem[];
    float* warp_results = shared_mem;
    float* block_result = &shared_mem[BLOCK_SIZE / WARP_SIZE];
    
    float thread_sum = 0.0f;
    
    // Process 4 elements at a time using vectorized loads
    if (global_thread_id < n_vec) {
        float4 log_pred4 = log_predictions_vec[global_thread_id];
        float4 target4 = targets_vec[global_thread_id];
        
        // Process vector elements
        thread_sum += expf(log_pred4.x) - target4.x * log_pred4.x;
        thread_sum += expf(log_pred4.y) - target4.y * log_pred4.y;
        thread_sum += expf(log_pred4.z) - target4.z * log_pred4.z;
        thread_sum += expf(log_pred4.w) - target4.w * log_pred4.w;
    }
    
    // First level reduction within warps
    thread_sum = warp_reduce(thread_sum);
    
    // Store warp results
    if (lane == 0) {
        warp_results[wid] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction within block
    if (wid == 0) {
        float warp_sum = (lane < (BLOCK_SIZE / WARP_SIZE)) ? warp_results[lane] : 0.0f;
        warp_sum = warp_reduce(warp_sum);
        
        if (lane == 0) {
            block_result[0] = warp_sum;
        }
    }
    __syncthreads();
    
    // Only one thread per block writes to global memory
    if (tid == 0) {
        block_results[blockIdx.x] = block_result[0];
    }
}

__global__ void final_reduction_kernel(
    float* __restrict__ block_results,
    float* __restrict__ output,
    const int num_blocks) {
    
    extern __shared__ float shared_mem[];
    const int tid = threadIdx.x;
    
    float sum = 0.0f;
    for (int i = tid; i < num_blocks; i += BLOCK_SIZE) {
        sum += block_results[i];
    }
    
    // Warp reduction
    sum = warp_reduce(sum);
    
    if (tid % WARP_SIZE == 0) {
        shared_mem[tid / WARP_SIZE] = sum;
    }
    __syncthreads();
    
    if (tid < (BLOCK_SIZE / WARP_SIZE)) {
        float final_sum = shared_mem[tid];
        final_sum = warp_reduce(final_sum);
        
        if (tid == 0) {
            *output = final_sum;
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    const int n_vec = n / 4;
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Calculate grid dimensions
    const int blocks = (n_vec + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate temporary storage for block results
    auto block_results = torch::empty({blocks}, log_predictions.options());
    
    // Shared memory size
    const int shared_mem_size = (BLOCK_SIZE / WARP_SIZE + 1) * sizeof(float);
    
    // Launch kernels
    kl_div_kernel_reduced_atomics<<<blocks, BLOCK_SIZE, shared_mem_size>>>(
        reinterpret_cast<const float4*>(log_predictions.data_ptr<float>()),
        reinterpret_cast<const float4*>(targets.data_ptr<float>()),
        block_results.data_ptr<float>(),
        n_vec
    );
    
    final_reduction_kernel<<<1, BLOCK_SIZE, BLOCK_SIZE / WARP_SIZE * sizeof(float)>>>(
        block_results.data_ptr<float>(),
        output.data_ptr<float>(),
        blocks
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA reduced atomics)");
}