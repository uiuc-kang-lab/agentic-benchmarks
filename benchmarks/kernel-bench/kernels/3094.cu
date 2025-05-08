#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// Warp-level reduction for maximum using shuffle
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum using shuffle
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized Softmax Kernel using shared memory for intra-block reduction and warp-level primitives for final stages
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Pointers for current batch row
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    // Shared memory is partitioned into two arrays: one for warp maxes and one for warp sums
    extern __shared__ float shared[];
    float* warp_maxes = shared;                  // Size: num_warps floats
    float* warp_sums = shared + num_warps;         // Size: num_warps floats

    // Phase 1: Compute the maximum value in the row
    float partial_max = -INFINITY;
    for (int i = tid; i < num_features; i += blockDim.x) {
        partial_max = fmaxf(partial_max, x_row[i]);
    }
    // Intra-warp reduction using warp shuffle
    float warp_max = warpReduceMax(partial_max);
    if (lane_id == 0) {
        warp_maxes[warp_id] = warp_max;
    }
    __syncthreads();

    // Final reduction for maximum: first warp reduces the partial warp max values
    float block_max = -INFINITY;
    if (tid < num_warps) {
        block_max = warp_maxes[tid];
    }
    block_max = warpReduceMax(block_max);
    if (tid == 0) {
        warp_maxes[0] = block_max;
    }
    __syncthreads();
    block_max = warp_maxes[0];

    // Phase 2: Compute exponentials and partial sum for softmax normalization
    float partial_sum = 0.0f;
    for (int i = tid; i < num_features; i += blockDim.x) {
        float exp_val = __expf(x_row[i] - block_max);
        y_row[i] = exp_val;  // Store intermediate exponential values
        partial_sum += exp_val;
    }
    // Intra-warp reduction for sum using shuffle
    float warp_sum = warpReduceSum(partial_sum);
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    // Final reduction for sum: first warp reduces the partial sums
    float block_sum = 0.0f;
    if (tid < num_warps) {
        block_sum = warp_sums[tid];
    }
    block_sum = warpReduceSum(block_sum);
    if (tid == 0) {
        warp_sums[0] = block_sum;
    }
    __syncthreads();
    float inv_sum = 1.0f / warp_sums[0];

    // Phase 3: Normalize the exponentials to compute softmax
    for (int i = tid; i < num_features; i += blockDim.x) {
        y_row[i] *= inv_sum;
    }
}

// Host function to launch the CUDA softmax kernel
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid_dim(batch_size);
    dim3 block_dim(THREADS_PER_BLOCK);
    int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    int shared_mem_size = sizeof(float) * num_warps * 2; // Two arrays: one for maxes and one for sums
    softmax_kernel<<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
}

// C++ interface exposed to PyTorch
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    int batch_size = x.size(0);
    int num_features = x.size(1);
    auto y = torch::empty_like(x);
    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, num_features);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp and Shared Memory Optimized Softmax (CUDA)");
}
