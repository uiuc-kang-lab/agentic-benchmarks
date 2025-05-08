#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Define warp size
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Kernel 1: Compute swish activation and perform intra-block reduction using shared memory and warp-level primitives
__global__ void swish_reduce_kernel(const float* __restrict__ x,
                                      float* __restrict__ y,
                                      float* __restrict__ blockSums,
                                      int64_t n) {
    extern __shared__ float sdata[]; // Shared memory to store warp-level sums
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;
    float local_sum = 0.0f;

    // Grid-stride loop: each thread processes multiple elements
    int stride = blockDim.x * gridDim.x;
    for (int i = globalIdx; i < n; i += stride) {
        float val = x[i];
        // Compute swish activation: y = x * sigmoid(x)
        float swish_val = val * (1.0f / (1.0f + expf(-val)));
        y[i] = swish_val;
        local_sum += swish_val;
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Each warp's lane 0 writes its reduced sum to shared memory
    int lane = tid % WARP_SIZE;
    if (lane == 0) {
        sdata[tid / WARP_SIZE] = local_sum;
    }
    __syncthreads();

    // Let first warp finish reduction of warp sums stored in shared memory
    float block_sum = 0.0f;
    int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    if (tid < numWarps) {
        block_sum = sdata[tid];
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
    }

    // Thread 0 writes the block's total sum to global memory
    if (tid == 0) {
        blockSums[blockIdx.x] = block_sum;
    }
}

// Kernel 2: Final reduction kernel to combine block sums into a single scalar
__global__ void final_reduction_kernel(const float* __restrict__ blockSums,
                                         float* __restrict__ sum_out,
                                         int numBlocks) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // Load blockSums into shared memory (if tid >= numBlocks, use 0)
    float sum = (tid < numBlocks) ? blockSums[tid] : 0.0f;
    sdata[tid] = sum;
    __syncthreads();

    // Standard reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *sum_out = sdata[0];
    }
}

// Forward function: Computes swish activation and returns both the activated tensor and the reduction sum
std::vector<torch::Tensor> swish_reduce_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");

    // Allocate output tensor with the same shape as input
    auto y = torch::empty_like(x);
    int64_t n = x.numel();

    // Configure kernel launch parameters
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // Allocate temporary tensor for block sums (one float per block)
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto blockSums = torch::empty({blocks}, options);
    // Allocate tensor for final reduction result (scalar)
    auto sum_tensor = torch::empty({1}, options);

    // Shared memory: one float per warp in the block
    int sharedMemSize = ((threads + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);

    // Launch kernel to compute swish activation and per-block reductions
    swish_reduce_kernel<<<blocks, threads, sharedMemSize>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        blockSums.data_ptr<float>(),
        n
    );

    // Determine number of threads for the final reduction kernel (next power of 2 >= blocks)
    int finalThreads = 1;
    while (finalThreads < blocks) {
        finalThreads *= 2;
    }

    // Launch final reduction kernel with one block
    final_reduction_kernel<<<1, finalThreads, finalThreads * sizeof(float)>>>(
        blockSums.data_ptr<float>(),
        sum_tensor.data_ptr<float>(),
        blocks
    );

    // Return both the swish activated tensor and the reduction sum as a vector of tensors
    return {y, sum_tensor};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_reduce_forward, "Swish activation forward pass with optimized reduction (CUDA)");
}
