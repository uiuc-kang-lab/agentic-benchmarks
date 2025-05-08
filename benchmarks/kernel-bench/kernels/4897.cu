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
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & (warpSize - 1);
    int warpId = tid >> 5;  // equivalent to tid / warpSize
    float sum = 0.0f;

    // Each thread processes multiple elements in the row
    for (int col = tid; col < D; col += blockDim.x) {
        sum += fabsf(x[row * D + col]);
    }

    // Intra-warp reduction using shuffle primitives
    sum = warpReduceSum(sum);

    // Use minimal shared memory to store each warp's sum
    __shared__ float warpSums[32]; // supports up to 1024 threads per block
    if (lane == 0) {
        warpSums[warpId] = sum;
    }

    __syncthreads();

    // Final reduction: first warp reduces the warp sums using shuffle intrinsics
    if (tid < warpSize) {
        int nWarps = (blockDim.x + warpSize - 1) / warpSize;
        float blockSum = (tid < nWarps) ? warpSums[tid] : 0.0f;
        blockSum = warpReduceSum(blockSum);
        if (tid == 0) {
            // Prevent division by zero
            warpSums[0] = (blockSum == 0.0f) ? 1e-12f : blockSum;
        }
    }

    __syncthreads();
    float total = warpSums[0];

    // Normalize each element of the row
    for (int col = tid; col < D; col += blockDim.x) {
        int idx = row * D + col;
        out[idx] = x[idx] / total;
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
