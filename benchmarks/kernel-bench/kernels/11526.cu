#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for configuration
__constant__ int d_num_elements;
__constant__ float d_scale_factor;

// Initialize constant memory
void init_constants(int n) {
    float scale = 1.0f / static_cast<float>(n);
    cudaMemcpyToSymbol(d_num_elements, &n, sizeof(int));
    cudaMemcpyToSymbol(d_scale_factor, &scale, sizeof(float));
}

__global__ void kl_div_constant_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_threads = blockDim.x;
    const int warp_size = 32;
    
    // Ensure aligned access within warps
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;
    const int aligned_idx = bid * num_threads + warp_id * warp_size + lane_id;
    
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];
    float thread_sum = 0.0f;
    
    // Process elements with stride of complete warps
    for (int i = aligned_idx; i < d_num_elements; i += gridDim.x * num_threads) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        thread_sum += __expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // Write warp results to shared memory
    if (lane_id == 0) {
        partial_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (tid < warp_size) {
        thread_sum = (tid < (blockDim.x / warp_size)) ? partial_sums[tid] : 0.0f;
        
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        
        if (tid == 0) {
            atomicAdd(output, thread_sum * d_scale_factor);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    
    // Initialize constant memory
    init_constants(n);
    
    // Create output tensor
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);
    
    kl_div_constant_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>()
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}