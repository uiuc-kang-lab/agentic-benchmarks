#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Constant memory for frequently accessed values
__constant__ float THRESHOLD = 1.0f;
__constant__ float HALF = 0.5f;
__constant__ float NORM_FACTOR;  // Will store 1.0f/n_elements
__constant__ int NUM_ELEMENTS;

__device__ __forceinline__ float compute_loss(float diff) {
    float abs_diff = fabsf(diff);
    return (abs_diff < THRESHOLD) ? HALF * diff * diff : abs_diff - HALF;
}

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void smooth_l1_loss_const_tuned_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output
) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int stride = gridDim.x * blockDim.x;
    const int lane = tid % warpSize;
    const int wid = tid / warpSize;
    
    float thread_sum = 0.0f;
    
    // Vectorized processing using float4
    const int vec_elements = NUM_ELEMENTS / 4;
    const float4* pred4 = reinterpret_cast<const float4*>(predictions);
    const float4* targ4 = reinterpret_cast<const float4*>(targets);
    
    #pragma unroll 2
    for (int i = gid; i < vec_elements; i += stride) {
        float4 p = __ldg(pred4 + i);
        float4 t = __ldg(targ4 + i);
        
        thread_sum += compute_loss(p.x - t.x);
        thread_sum += compute_loss(p.y - t.y);
        thread_sum += compute_loss(p.z - t.z);
        thread_sum += compute_loss(p.w - t.w);
    }
    
    // Handle remaining elements
    const int remainder_start = vec_elements * 4;
    #pragma unroll 4
    for (int i = remainder_start + gid; i < NUM_ELEMENTS; i += stride) {
        float diff = __ldg(predictions + i) - __ldg(targets + i);
        thread_sum += compute_loss(diff);
    }
    
    // Apply normalization factor early to reduce final atomic add impact
    thread_sum *= NORM_FACTOR;
    
    // Warp-level reduction
    thread_sum = warp_reduce(thread_sum);
    
    __shared__ float warp_sums[32];
    if (lane == 0) {
        warp_sums[wid] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction using first warp
    if (wid == 0) {
        float val = (tid < (blockDim.x + warpSize - 1) / warpSize) ? warp_sums[lane] : 0.0f;
        val = warp_reduce(val);
        if (lane == 0) {
            atomicAdd(output, val);
        }
    }
}

torch::Tensor smooth_l1_loss_const_tuned(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "Inputs must be CUDA tensors");
    
    const int n_elements = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());
    
    // Update constant memory with current n_elements and normalization factor
    float norm_factor = 1.0f / n_elements;
    cudaMemcpyToSymbol(NORM_FACTOR, &norm_factor, sizeof(float));
    cudaMemcpyToSymbol(NUM_ELEMENTS, &n_elements, sizeof(int));
    
    const int block_size = 256;
    const int vec_elements = n_elements / 4;
    const int grid_size = std::min(65535, (vec_elements + block_size - 1) / block_size);
    
    smooth_l1_loss_const_tuned_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>()
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_const_tuned, "Constant memory optimized Smooth L1 Loss (CUDA)");
}