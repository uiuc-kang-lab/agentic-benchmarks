#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Optimized warp-level reduction using shuffle intrinsics
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void l1_norm_hybrid_kernel(const float* __restrict__ x,
                                    float* __restrict__ out,
                                    const int N,
                                    const int D) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int step = blockDim.x;
    const int lane = tid & (warpSize-1);
    const int wid = tid / warpSize;

    // Step 1: Compute local sum with vectorized loads when possible
    float sum = 0.0f;
    if (D >= 4) {
        const float4* x4 = reinterpret_cast<const float4*>(x + row * D);
        for (int col = tid; col < D/4; col += step) {
            float4 vals = x4[col];
            sum += fabsf(vals.x) + fabsf(vals.y) + fabsf(vals.z) + fabsf(vals.w);
        }
        // Handle remaining elements
        for (int col = tid + (D/4)*4; col < D; col += step) {
            sum += fabsf(x[row * D + col]);
        }
    } else {
        for (int col = tid; col < D; col += step) {
            sum += fabsf(x[row * D + col]);
        }
    }

    // Step 2: Warp-level reduction
    sum = warpReduceSum(sum);

    // Step 3: Store warp results to shared memory
    if (lane == 0) {
        sdata[wid] = sum;
    }
    __syncthreads();

    // Step 4: Final reduction with first warp
    if (tid < 32) {
        float warp_sum = (tid < (step/warpSize)) ? sdata[tid] : 0.0f;
        warp_sum = warpReduceSum(warp_sum);
        
        if (tid == 0) {
            warp_sum = (warp_sum == 0.0f) ? 1e-12f : warp_sum;
            sdata[0] = warp_sum;
        }
    }
    __syncthreads();
    
    const float total = sdata[0];

    // Step 5: Normalize with vectorized stores when possible
    if (D >= 4) {
        float4* out4 = reinterpret_cast<float4*>(out + row * D);
        const float4* x4 = reinterpret_cast<const float4*>(x + row * D);
        for (int col = tid; col < D/4; col += step) {
            float4 vals = x4[col];
            vals.x /= total;
            vals.y /= total;
            vals.z /= total;
            vals.w /= total;
            out4[col] = vals;
        }
        // Handle remaining elements
        for (int col = tid + (D/4)*4; col < D; col += step) {
            out[row * D + col] = x[row * D + col] / total;
        }
    } else {
        for (int col = tid; col < D; col += step) {
            out[row * D + col] = x[row * D + col] / total;
        }
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor.");
    x = x.contiguous();

    auto out = torch::empty_like(x);
    const int N = x.size(0);
    const int D = x.size(1);
    const int threads = std::min<int>(1024, ((D + 3)/4) * 4); // Align with vector loads
    const int shared_mem_size = (threads/32) * sizeof(float);

    l1_norm_hybrid_kernel<<<N, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, D
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization forward pass (CUDA with hybrid optimizations)");
}