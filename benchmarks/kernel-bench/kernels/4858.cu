#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void l1_norm_coalesced_kernel(const float* __restrict__ x,
                                        float* __restrict__ out,
                                        const int N,
                                        const int D) {
    __shared__ float warp_sums[BLOCK_SIZE/WARP_SIZE];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    // Ensure coalesced memory access by having consecutive threads
    // access consecutive memory locations within a row
    float thread_sum = 0.0f;
    
    if (D >= 4) {
        // Use vectorized loads for better memory throughput
        const int vec_elements = D / 4;
        const float4* x_vec = reinterpret_cast<const float4*>(x + row * D);
        
        // Stride by number of threads to maintain coalescing
        #pragma unroll 4
        for (int idx = tid; idx < vec_elements; idx += blockDim.x) {
            float4 data = __ldg(&x_vec[idx]);
            thread_sum += fabsf(data.x) + fabsf(data.y) + 
                         fabsf(data.z) + fabsf(data.w);
        }
        
        // Handle remaining elements
        const int remainder_start = vec_elements * 4;
        for (int idx = remainder_start + tid; idx < D; idx += blockDim.x) {
            thread_sum += fabsf(__ldg(&x[row * D + idx]));
        }
    } else {
        // For small D, use regular loads while maintaining coalescing
        for (int idx = tid; idx < D; idx += blockDim.x) {
            thread_sum += fabsf(__ldg(&x[row * D + idx]));
        }
    }
    
    // Warp-level reduction
    thread_sum = warp_reduce_sum(thread_sum);
    
    // First thread in each warp writes the result
    if (lane == 0) {
        warp_sums[wid] = thread_sum;
    }
    __syncthreads();
    
    // First warp reduces the warp sums
    if (wid == 0) {
        thread_sum = (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) 
                    ? warp_sums[lane] : 0.0f;
        thread_sum = warp_reduce_sum(thread_sum);
        
        if (lane == 0) {
            // Avoid division by zero
            warp_sums[0] = (thread_sum == 0.0f) ? 1e-12f : thread_sum;
        }
    }
    __syncthreads();
    
    const float row_sum = warp_sums[0];
    
    // Normalize with coalesced access pattern
    if (D >= 4) {
        const int vec_elements = D / 4;
        float4* out_vec = reinterpret_cast<float4*>(out + row * D);
        const float4* x_vec = reinterpret_cast<const float4*>(x + row * D);
        
        #pragma unroll 4
        for (int idx = tid; idx < vec_elements; idx += blockDim.x) {
            float4 data = __ldg(&x_vec[idx]);
            data.x /= row_sum;
            data.y /= row_sum;
            data.z /= row_sum;
            data.w /= row_sum;
            out_vec[idx] = data;
        }
        
        // Handle remaining elements
        const int remainder_start = vec_elements * 4;
        for (int idx = remainder_start + tid; idx < D; idx += blockDim.x) {
            out[row * D + idx] = x[row * D + idx] / row_sum;
        }
    } else {
        for (int idx = tid; idx < D; idx += blockDim.x) {
            out[row * D + idx] = x[row * D + idx] / row_sum;
        }
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor.");
    x = x.contiguous();
    
    const int N = x.size(0);
    const int D = x.size(1);
    
    auto out = torch::empty_like(x);
    
    // Use fixed block size for better occupancy
    const int threads = BLOCK_SIZE;
    
    l1_norm_coalesced_kernel<<<N, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced L1 Normalization forward (CUDA)");
}