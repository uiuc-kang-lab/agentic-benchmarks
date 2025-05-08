#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void l1_norm_kernel_optimized(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       float* __restrict__ row_sums,
                                       const int N,
                                       const int D) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    // Each thread accumulates its portion in registers
    float local_sum = 0.0f;
    
    if (4 * tid < D) {  // Check if we can use vectorized loads
        const float4* x_vec = reinterpret_cast<const float4*>(x + row * D);
        const int vec_elements = D / 4;
        
        #pragma unroll 4
        for (int i = tid; i < vec_elements; i += blockDim.x) {
            float4 vals = __ldg(&x_vec[i]);
            local_sum += fabsf(vals.x) + fabsf(vals.y) + fabsf(vals.z) + fabsf(vals.w);
        }
        
        // Handle remaining elements
        const int remainder = D % 4;
        if (tid < remainder) {
            local_sum += fabsf(__ldg(&x[row * D + D - remainder + tid]));
        }
    } else {
        // Scalar loads for small D or remainder
        for (int i = tid; i < D; i += blockDim.x) {
            local_sum += fabsf(__ldg(&x[row * D + i]));
        }
    }
    
    // Warp-level reduction
    float warp_sum = warp_reduce_sum(local_sum);
    
    // Write warp results to shared memory
    if (lane == 0) {
        smem[wid] = warp_sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (wid == 0) {
        float final_sum = (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? smem[lane] : 0.0f;
        final_sum = warp_reduce_sum(final_sum);
        
        // Single atomic operation per row
        if (lane == 0) {
            if (final_sum == 0.0f) final_sum = 1e-12f;
            row_sums[row] = final_sum;
        }
    }
    __syncthreads();
    
    // Normalize using the computed sum
    const float row_sum = row_sums[row];
    
    if (4 * tid < D) {
        float4* out_vec = reinterpret_cast<float4*>(out + row * D);
        const float4* x_vec = reinterpret_cast<const float4*>(x + row * D);
        const int vec_elements = D / 4;
        
        #pragma unroll 4
        for (int i = tid; i < vec_elements; i += blockDim.x) {
            float4 vals = __ldg(&x_vec[i]);
            vals.x /= row_sum;
            vals.y /= row_sum;
            vals.z /= row_sum;
            vals.w /= row_sum;
            out_vec[i] = vals;
        }
        
        // Handle remaining elements
        const int remainder = D % 4;
        if (tid < remainder) {
            const int idx = row * D + D - remainder + tid;
            out[idx] = x[idx] / row_sum;
        }
    } else {
        for (int i = tid; i < D; i += blockDim.x) {
            out[row * D + i] = x[row * D + i] / row_sum;
        }
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(x.dim() == 2, "Input must be a 2D tensor");
    
    const int N = x.size(0);
    const int D = x.size(1);
    
    auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
    auto out = torch::empty_like(x);
    auto row_sums = torch::empty({N}, options);
    
    const int threads = std::min(256, ((D + 3) / 4) * 4);  // Align with vector loads
    const int warps_per_block = (threads + WARP_SIZE - 1) / WARP_SIZE;
    const int smem_size = warps_per_block * sizeof(float);
    
    l1_norm_kernel_optimized<<<N, threads, smem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        row_sums.data_ptr<float>(),
        N,
        D
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization forward (CUDA)");
}