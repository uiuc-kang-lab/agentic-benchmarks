#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Use constant memory for frequently accessed data
__constant__ int D_const;

// Warp-level reduction using shuffle intrinsics
__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void l1_norm_constant_kernel(const float* __restrict__ x,
                                         float* __restrict__ out,
                                         int N) {
    int row = blockIdx.x;
    int row_offset = row * D_const;
    float local_sum = 0.0f;

    // Process as many elements as possible using 128-bit (float4) vectorized loads
    int nvec = D_const / 4;
    int rem = D_const % 4;

    // Pointer for vectorized loads; we assume x is contiguous and appropriately aligned
    const float4* x_vec = reinterpret_cast<const float4*>(x + row_offset);

    // Each thread loads consecutive vectorized elements, ensuring coalesced access
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
        float4 v = __ldg(x_vec + i);
        local_sum += fabsf(v.x) + fabsf(v.y) + fabsf(v.z) + fabsf(v.w);
    }

    // Process any remaining elements for this row
    int base = nvec * 4;
    for (int j = threadIdx.x; j < rem; j += blockDim.x) {
        local_sum += fabsf(x[row_offset + base + j]);
    }

    // Perform warp-level reduction
    float warp_sum = warpReduceSum(local_sum);

    // Use shared memory to reduce across warps within the block
    extern __shared__ float shared[];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        shared[warp_id] = warp_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    int numWarps = (blockDim.x + 31) / 32;
    if (threadIdx.x < numWarps) {
        block_sum = shared[lane];
    }
    block_sum = warpReduceSum(block_sum);

    if (threadIdx.x == 0) {
        shared[0] = (block_sum == 0.0f ? 1e-12f : block_sum);
    }
    __syncthreads();
    float norm = shared[0];

    float4* out_vec = reinterpret_cast<float4*>(out + row_offset);
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
        float4 v = __ldg(x_vec + i);
        float4 res;
        res.x = v.x / norm;
        res.y = v.y / norm;
        res.z = v.z / norm;
        res.w = v.w / norm;
        out_vec[i] = res;
    }
    for (int j = threadIdx.x; j < rem; j += blockDim.x) {
        int idx = row_offset + base + j;
        out[idx] = x[idx] / norm;
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    x = x.contiguous();
    
    auto out = torch::empty_like(x);
    int N = x.size(0);
    int D = x.size(1);

    // Set constant memory value for dimension
    cudaMemcpyToSymbol(D_const, &D, sizeof(int));

    int warp_size = 32;
    int threads = std::min<int>(1024, (D + 3) / 4 * warp_size);
    threads = std::min<int>((threads + warp_size - 1) / warp_size * warp_size, 1024);
    int shared_mem = (threads / warp_size) * sizeof(float);

    l1_norm_constant_kernel<<<N, threads, shared_mem>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization with constant memory optimization");
}