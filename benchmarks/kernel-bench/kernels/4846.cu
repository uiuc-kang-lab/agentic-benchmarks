#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void l1_norm_forward_kernel_optimized(const float* __restrict__ x,
                                               float* __restrict__ out,
                                               const int N,
                                               const int D) {
    __shared__ float s_partial_sums[BLOCK_SIZE];
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid & (WARP_SIZE-1);
    const int warp_id = tid >> 5;
    
    float thread_sum = 0.0f;
    
    if (D >= 4) {
        const int vec_elements = D >> 2;
        const float4* x_vec = reinterpret_cast<const float4*>(x + row * D);
        
        #pragma unroll 4
        for (int i = tid; i < vec_elements; i += blockDim.x) {
            float4 values = __ldg(&x_vec[i]);
            thread_sum += fabsf(values.x) + fabsf(values.y) + 
                         fabsf(values.z) + fabsf(values.w);
        }
        
        const int remainder_start = vec_elements << 2;
        for (int i = remainder_start + tid; i < D; i += blockDim.x) {
            thread_sum += fabsf(__ldg(&x[row * D + i]));
        }
    } else {
        for (int i = tid; i < D; i += blockDim.x) {
            thread_sum += fabsf(__ldg(&x[row * D + i]));
        }
    }
    
    thread_sum = warpReduceSum(thread_sum);
    
    if (lane_id == 0) {
        s_partial_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    if (warp_id == 0 && lane_id < (blockDim.x / WARP_SIZE)) {
        float warp_sum = s_partial_sums[lane_id];
        warp_sum = warpReduceSum(warp_sum);
        
        if (lane_id == 0) {
            s_partial_sums[0] = (warp_sum > 0.0f) ? warp_sum : 1e-12f;
        }
    }
    __syncthreads();
    
    const float norm = s_partial_sums[0];
    
    if (D >= 4) {
        const int vec_elements = D >> 2;
        float4* out_vec = reinterpret_cast<float4*>(out + row * D);
        const float4* x_vec = reinterpret_cast<const float4*>(x + row * D);
        
        #pragma unroll 4
        for (int i = tid; i < vec_elements; i += blockDim.x) {
            float4 values = __ldg(&x_vec[i]);
            values.x /= norm;
            values.y /= norm;
            values.z /= norm;
            values.w /= norm;
            out_vec[i] = values;
        }
        
        const int remainder_start = vec_elements << 2;
        for (int i = remainder_start + tid; i < D; i += blockDim.x) {
            out[row * D + i] = __ldg(&x[row * D + i]) / norm;
        }
    } else {
        for (int i = tid; i < D; i += blockDim.x) {
            out[row * D + i] = __ldg(&x[row * D + i]) / norm;
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
    
    l1_norm_forward_kernel_optimized<<<N, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized L1 Normalization forward (CUDA)");
}