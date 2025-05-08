#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for kernel configuration
__constant__ int d_block_size = 256;
__constant__ int d_vector_size = 4;
__constant__ float d_eps = 1e-8;

template<unsigned int BLOCK_SIZE>
__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float4 load_vector(const float* __restrict__ ptr, int idx) {
    float4 v;
    v.x = ptr[idx];
    v.y = ptr[idx + 1];
    v.z = ptr[idx + 2];
    v.w = ptr[idx + 3];
    return v;
}

__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

template<unsigned int BLOCK_SIZE>
__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    extern __shared__ float sdata[];
    
    const unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * d_block_size * d_vector_size + tid * d_vector_size;
    const unsigned int grid_stride = d_block_size * d_vector_size * gridDim.x;
    
    float thread_sum = 0.0f;
    
    // Process 4 elements at a time using vectorized loads
    while (i + 3 < n) {
        float4 log_preds = load_vector(log_predictions, i);
        float4 targs = load_vector(targets, i);
        
        thread_sum += compute_kl_div(log_preds.x, targs.x);
        thread_sum += compute_kl_div(log_preds.y, targs.y);
        thread_sum += compute_kl_div(log_preds.z, targs.z);
        thread_sum += compute_kl_div(log_preds.w, targs.w);
        
        i += grid_stride;
    }
    
    // Handle remaining elements
    while (i < n) {
        thread_sum += compute_kl_div(log_predictions[i], targets[i]);
        i++;
    }
    
    // Store in shared memory
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Reduction in shared memory
    if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    
    // Final warp reduction
    if (tid < 32) {
        float warp_sum = sdata[tid];
        if (BLOCK_SIZE >= 32) warp_sum += sdata[tid + 32];
        warp_sum = warp_reduce<BLOCK_SIZE>(warp_sum);
        
        if (tid == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Copy configuration to constant memory
    const int block_size = 256;
    const int vector_size = 4;
    const float eps = 1e-8f;
    cudaMemcpyToSymbol(d_block_size, &block_size, sizeof(int));
    cudaMemcpyToSymbol(d_vector_size, &vector_size, sizeof(int));
    cudaMemcpyToSymbol(d_eps, &eps, sizeof(float));
    
    const int blocks = min((n + block_size * vector_size - 1) / (block_size * vector_size), 1024);
    const int shared_mem = block_size * sizeof(float);
    
    kl_div_kernel<256><<<blocks, block_size, shared_mem>>>(
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