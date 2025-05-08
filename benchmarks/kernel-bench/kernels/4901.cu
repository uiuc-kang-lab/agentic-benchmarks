#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Warp-level reduction using shuffle intrinsics
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// CUDA kernel for L1 normalization using warp-level primitives for reduction
__global__ void l1_norm_forward_kernel_warp(const float* __restrict__ x,
                                               float* __restrict__ out,
                                               int N,
                                               int D) {
    // Each block processes one row
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & (warpSize - 1);
    const int warpId = tid >> 5;  // equivalent to tid / warpSize
    
    // Use vectorized loads where possible for better memory coalescing
    float sum = 0.0f;
    const int row_offset = row * D;
    
    // Each thread processes multiple elements using vectorized loads where possible
    if (D >= 4 && tid * 4 < D) {
        const float4* x_vec = reinterpret_cast<const float4*>(x + row_offset);
        for (int col = tid * 4; col < D - 3; col += blockDim.x * 4) {
            float4 val = x_vec[col >> 2];
            sum += fabsf(val.x) + fabsf(val.y) + fabsf(val.z) + fabsf(val.w);
        }
        
        // Handle remaining elements
        for (int col = tid + ((D >> 2) << 2); col < D; col += blockDim.x) {
            sum += fabsf(x[row_offset + col]);
        }
    } else {
        // Fall back to regular loading for small D
        for (int col = tid; col < D; col += blockDim.x) {
            sum += fabsf(x[row_offset + col]);
        }
    }

    // Intra-warp reduction using shuffle primitives
    sum = warpReduceSum(sum);

    // Use minimal shared memory with padding to avoid bank conflicts
    __shared__ float warpSums[33]; // 33 instead of 32 to avoid bank conflicts
    if (lane == 0) {
        warpSums[warpId] = sum;
    }

    __syncthreads();

    // Final reduction: first warp reduces the warp sums using shuffle intrinsics
    if (tid < warpSize) {
        const int nWarps = (blockDim.x + warpSize - 1) / warpSize;
        float blockSum = (tid < nWarps) ? warpSums[tid] : 0.0f;
        blockSum = warpReduceSum(blockSum);
        if (tid == 0) {
            // Prevent division by zero
            warpSums[0] = (blockSum == 0.0f) ? 1e-12f : blockSum;
        }
    }

    __syncthreads();
    const float total = warpSums[0];
    const float inv_total = __fdividef(1.0f, total); // Fast division

    // Normalize each element using vectorized stores where possible
    if (D >= 4 && tid * 4 < D) {
        float4* out_vec = reinterpret_cast<float4*>(out + row_offset);
        const float4* x_vec = reinterpret_cast<const float4*>(x + row_offset);
        
        for (int col = tid * 4; col < D - 3; col += blockDim.x * 4) {
            float4 val = x_vec[col >> 2];
            val.x *= inv_total;
            val.y *= inv_total;
            val.z *= inv_total;
            val.w *= inv_total;
            out_vec[col >> 2] = val;
        }
        
        // Handle remaining elements
        for (int col = tid + ((D >> 2) << 2); col < D; col += blockDim.x) {
            out[row_offset + col] = x[row_offset + col] * inv_total;
        }
    } else {
        // Fall back to regular storing for small D
        for (int col = tid; col < D; col += blockDim.x) {
            out[row_offset + col] = x[row_offset + col] * inv_total;
        }
    }
}

// Forward function exposed to PyTorch via pybind11
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor.");
    x = x.contiguous();

    auto out = torch::empty_like(x);
    int N = x.size(0);
    int D = x.size(1);
    
    // Choose an appropriate block size; must be a multiple of warpSize
    int blockSize = (D < 256) ? D : 256;

    l1_norm_forward_kernel_warp<<<N, blockSize>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, D
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization forward pass (CUDA with warp-level primitives)");
}
