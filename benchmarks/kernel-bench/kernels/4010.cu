#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel 1: Compute element-wise ELU activation and accumulate a local sum per thread using a grid-stride loop.
// Then perform an intra-block reduction using warp-level primitives and shared memory.
__global__ void elu_activation_reduce_kernel(const float* __restrict__ x,
                                              float* __restrict__ out,
                                              float alpha,
                                              int n,
                                              float* __restrict__ block_sums) {
    float local_sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread (grid-stride loop):
    for (int i = idx; i < n; i += stride) {
        float val = x[i];
        float activated = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
        out[i] = activated;
        local_sum += activated;
    }

    // Intra-warp reduction using warp-level primitives (__shfl_down_sync):
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Use shared memory to reduce the sums from each warp within the block.
    __shared__ float shared[32]; // Maximum number of warps per block assumed <= 32
    int lane = threadIdx.x & 31;         // index within the warp
    int warpId = threadIdx.x >> 5;         // warp index in the block
    if (lane == 0) {
        shared[warpId] = local_sum;
    }
    __syncthreads();

    // Let the first warp finalize the reduction within the block
    if (threadIdx.x < (blockDim.x + 31) / 32) {
        local_sum = shared[lane];
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(mask, local_sum, offset);
        }
        if (lane == 0) {
            block_sums[blockIdx.x] = local_sum;
        }
    }
}

// Kernel 2: Final reduction kernel to aggregate per-block sums into a single scalar.
// This kernel uses shared memory with a binary tree reduction pattern combined with warp-level primitives.
__global__ void final_reduce_kernel(float* block_sums, int numElements) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    // Load block_sums into shared memory
    float sum = (tid < numElements) ? block_sums[tid] : 0.0f;
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory (binary tree reduction)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write the final result back to block_sums[0]
    if (tid == 0) {
        block_sums[0] = sdata[0];
    }
}

// Host interface: This function performs ELU activation on input tensor 'x' and simultaneously computes
// the sum of all activated elements using an in-kernel reduction. It returns a tuple of the activation tensor
// and a scalar tensor holding the reduction sum.
std::vector<torch::Tensor> elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // Allocate tensor for per-block sums on the same device as input
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto block_sums = torch::empty({blocks}, options);

    // Launch Kernel 1: Compute activation and per-block reduction sums
    elu_activation_reduce_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), alpha, n, block_sums.data_ptr<float>()
    );

    // Launch Kernel 2 only if there is more than one block to reduce the block_sums to a single value
    if (blocks > 1) {
        int finalThreads = 256;
        int sharedMemSize = finalThreads * sizeof(float);
        final_reduce_kernel<<<1, finalThreads, sharedMemSize>>>(block_sums.data_ptr<float>(), blocks);
    }

    // The final reduced sum is in block_sums[0]. We create a scalar tensor from it.
    auto reduction = block_sums.slice(0, 0, 1);

    // Return both the activated tensor and the reduction sum as a tuple
    return {out, reduction};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda, "ELU activation with fused reduction using shared memory and warp-level primitives (CUDA)");
}
