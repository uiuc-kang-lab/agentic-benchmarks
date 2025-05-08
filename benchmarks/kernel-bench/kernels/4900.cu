#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Warp-level reduction using shuffle intrinsics
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel for L1 normalization using shared memory for intra-block reduction
// and warp-level shuffle for final stage reduction.
__global__ void l1_norm_forward_kernel_shfl(const float* __restrict__ x,
                                              float* __restrict__ out,
                                              int N,
                                              int D) {
    extern __shared__ float sdata[]; // Shared memory for warp-level partial sums
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & (32 - 1);
    int warp_id = tid / 32;
    float sum = 0.0f;

    // Each thread processes a subset of elements in the row
    for (int col = tid; col < D; col += blockDim.x) {
        float val = x[row * D + col];
        sum += fabsf(val);
    }

    // Intra-warp reduction using shuffle
    sum = warpReduceSum(sum);

    // Write reduced sum of each warp to shared memory
    if (lane == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction: first warp loads warp sums from shared memory and reduces
    if (tid < (blockDim.x + 32 - 1) / 32) {
        float warp_sum = sdata[lane];
        warp_sum = warpReduceSum(warp_sum);
        if (tid == 0) {
            // Avoid division by zero
            sdata[0] = (warp_sum == 0.0f) ? 1e-12f : warp_sum;
        }
    }
    __syncthreads();

    float total_sum = sdata[0];

    // Normalize the row using the computed total sum
    for (int col = tid; col < D; col += blockDim.x) {
        out[row * D + col] = x[row * D + col] / total_sum;
    }
}

// Host function interfaced with PyTorch
// This function launches one block per row of the 2D tensor
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected a 2D tensor.");
    x = x.contiguous();

    auto out = torch::empty_like(x);
    int N = x.size(0);
    int D = x.size(1);
    
    // Choose number of threads per block (up to 1024, but at least D if smaller)
    int threads = std::min(1024, D);
    // Calculate number of warps required
    int numWarps = (threads + 32 - 1) / 32;
    int shared_mem_size = numWarps * sizeof(float);

    l1_norm_forward_kernel_shfl<<<N, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, D
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization forward pass (CUDA) using shared memory and warp-level reduction");
}
