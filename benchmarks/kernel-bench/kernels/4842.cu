#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32
#define BLOCK_SIZE 256  // Optimal block size for H100

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void l1_norm_balanced_kernel(const float* __restrict__ x,
                                      float* __restrict__ out,
                                      const int N,
                                      const int D) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    // Calculate number of elements per thread for balanced distribution
    const int elements_per_thread = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float local_sum = 0.0f;

    // Phase 1: Compute partial sums with coalesced memory access
    if (D >= 4 && (D % 4) == 0) {
        // Vectorized loads for aligned data
        const float4* x_vec = reinterpret_cast<const float4*>(x);
        const int vec_elements = D / 4;
        const int vec_per_thread = (vec_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        #pragma unroll 4
        for (int i = 0; i < vec_per_thread; i++) {
            const int idx = tid + i * BLOCK_SIZE;
            if (idx < vec_elements) {
                float4 data = __ldg(&x_vec[row * vec_elements + idx]);
                local_sum += fabsf(data.x) + fabsf(data.y) + 
                            fabsf(data.z) + fabsf(data.w);
            }
        }
    } else {
        // Scalar loads with balanced distribution
        #pragma unroll 4
        for (int i = 0; i < elements_per_thread; i++) {
            const int idx = tid + i * BLOCK_SIZE;
            if (idx < D) {
                local_sum += fabsf(__ldg(&x[row * D + idx]));
            }
        }
    }

    // Warp-level reduction
    local_sum = warpReduceSum(local_sum);
    
    // Write warp results to shared memory
    if (tid % WARP_SIZE == 0) {
        shared[tid / WARP_SIZE] = local_sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (tid < num_warps) {
        float warp_sum = (tid < num_warps) ? shared[tid] : 0.0f;
        warp_sum = warpReduceSum(warp_sum);
        
        if (tid == 0) {
            shared[0] = (warp_sum > 0.0f) ? warp_sum : 1e-12f;
        }
    }
    __syncthreads();
    
    const float norm = shared[0];

    // Phase 2: Normalize with balanced workload
    if (D >= 4 && (D % 4) == 0) {
        float4* out_vec = reinterpret_cast<float4*>(out);
        const float4* x_vec = reinterpret_cast<const float4*>(x);
        const int vec_elements = D / 4;
        const int vec_per_thread = (vec_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        #pragma unroll 4
        for (int i = 0; i < vec_per_thread; i++) {
            const int idx = tid + i * BLOCK_SIZE;
            if (idx < vec_elements) {
                float4 data = __ldg(&x_vec[row * vec_elements + idx]);
                data.x /= norm;
                data.y /= norm;
                data.z /= norm;
                data.w /= norm;
                out_vec[row * vec_elements + idx] = data;
            }
        }
    } else {
        #pragma unroll 4
        for (int i = 0; i < elements_per_thread; i++) {
            const int idx = tid + i * BLOCK_SIZE;
            if (idx < D) {
                out[row * D + idx] = __ldg(&x[row * D + idx]) / norm;
            }
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

    const int shared_mem_size = (BLOCK_SIZE / WARP_SIZE) * sizeof(float);
    
    l1_norm_balanced_kernel<<<N, BLOCK_SIZE, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced L1 Normalization forward (CUDA)");
}