#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Warp-level reduction using shuffle intrinsics
__device__ inline float warpReduceSum(float val) {
    // Assuming a warp has 32 threads
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// CUDA kernel that computes row-wise L1 normalization while dynamically adjusting block size
__global__ void l1_norm_dynamic_kernel(const float* __restrict__ x,
                                         float* __restrict__ out,
                                         int N,
                                         int D) {
    // Each block handles one row
    int row = blockIdx.x;
    int row_offset = row * D;
    float local_sum = 0.0f;

    // Process elements with stride-1 access pattern for better coalescing
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        local_sum += fabsf(x[row_offset + i]);
    }

    // Perform warp-level reduction
    float warp_sum = warpReduceSum(local_sum);

    // Use shared memory to reduce across warps within the block
    __shared__ float shared[32]; // enough for up to 32 warps per block
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5; 
    if (lane == 0) {
        shared[warp_id] = warp_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    // Let the first warp load the warp sums
    int numWarps = (blockDim.x + 31) / 32;
    if (threadIdx.x < numWarps) {
        block_sum = shared[lane];
    }
    block_sum = warpReduceSum(block_sum);

    // Thread 0 writes the final sum to shared memory for broadcasting
    if (threadIdx.x == 0) {
        // Avoid division by zero
        shared[0] = (block_sum == 0.0f ? 1e-12f : block_sum);
    }
    __syncthreads();
    float norm = shared[0];

    // Normalization: use vectorized stores to write results in a coalesced manner
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

// The forward function that sets up the grid and launches the kernel
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    x = x.contiguous();
    
    auto out = torch::empty_like(x);
    int N = x.size(0);
    int D = x.size(1);

    // Select optimal block size through empirical testing for this specific GPU setup
    int threads;
    if (D <= 64) {
        threads = 64;
    } else if (D <= 128) {
        threads = 128;
    } else if (D <= 256) {
        threads = 256;
    } else {
        threads = 512;
    }
    // Ensure we do not launch more threads than elements
    threads = std::min(threads, D);
    // Round up to a multiple of 32 (warp size)
    if (threads % 32 != 0) {
        threads = ((threads + 31) / 32) * 32;
        threads = std::min(threads, 1024);
    }

    int numWarps = threads / 32;
    int shared_mem = numWarps * sizeof(float);

    // Launch one block per row
    l1_norm_dynamic_kernel<<<N, threads, shared_mem>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, D
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization with dynamic block size optimization (CUDA)");
}