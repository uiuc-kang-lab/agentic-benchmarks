#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void l1_norm_forward_kernel_coalesced(const float* __restrict__ x,
                                                  float* __restrict__ out,
                                                  int N,
                                                  int D) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warpId = tid / warpSize; // Calculate the warp ID
    const int lane = tid % warpSize;
    const int warps_per_block = blockDim.x / warpSize;
    
    // Ensure aligned access to global memory
    float sum = 0.0f;
    const int vec_size = 4;
    const int aligned_offset = (row * D + tid) & ~(vec_size - 1);
    const int aligned_end = (row * D + D) & ~(vec_size - 1);
    
    // Process aligned elements using vectorized loads
    for (int i = aligned_offset; i < aligned_end; i += blockDim.x * vec_size) {
        if (i < (row + 1) * D) {
            float4 v = reinterpret_cast<const float4*>(x)[i / vec_size];
            sum += fabsf(v.x) + fabsf(v.y) + fabsf(v.z) + fabsf(v.w);
        }
    }
    
    // Handle remaining unaligned elements
    for (int i = aligned_end + tid; i < (row + 1) * D; i += blockDim.x) {
        sum += fabsf(x[i]);
    }
    
    // Warp-level reduction
    sum = warpReduceSum(sum);
    
    if (lane == 0) {
        sdata[warpId] = sum;
    }
    __syncthreads();
    
    // Final reduction using first warp
    if (warpId == 0) {
        sum = (tid < warps_per_block) ? sdata[tid] : 0.0f;
        sum = warpReduceSum(sum);
        
        if (lane == 0) {
            sdata[0] = (sum == 0.0f) ? 1e-12f : sum;
        }
    }
    __syncthreads();
    
    const float total_sum = sdata[0];
    
    // Normalize using aligned access
    for (int i = aligned_offset; i < aligned_end; i += blockDim.x * vec_size) {
        if (i < (row + 1) * D) {
            float4 v = reinterpret_cast<const float4*>(x)[i / vec_size];
            float4 result;
            result.x = v.x / total_sum;
            result.y = v.y / total_sum;
            result.z = v.z / total_sum;
            result.w = v.w / total_sum;
            reinterpret_cast<float4*>(out)[i / vec_size] = result;
        }
    }
    
    // Handle remaining unaligned elements
    for (int i = aligned_end + tid; i < (row + 1) * D; i += blockDim.x) {
        out[i] = x[i] / total_sum;
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor.");
    x = x.contiguous();

    auto out = torch::empty_like(x);
    int N = x.size(0);
    int D = x.size(1);
    
    int threads = 256;  // Optimal thread count for memory coalescing
    int warps_per_block = threads / warpSize;
    int shared_mem_size = warps_per_block * sizeof(float);
    
    l1_norm_forward_kernel_coalesced<<<N, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization forward pass (CUDA with coalesced memory access)");
}