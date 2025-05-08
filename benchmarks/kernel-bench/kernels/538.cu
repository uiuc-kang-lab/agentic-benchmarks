#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// This kernel performs a matrix-scalar multiplication (C = A * s) while simultaneously
// computing the sum of all multiplied elements using an efficient block-level reduction.
// It uses a grid-stride loop to process elements, warp-level primitives (__shfl_down_sync) for
// intra-warp reduction, and shared memory for intra-block reduction. The final block sums are
// atomically added to a global accumulator. This fused approach can be useful if a reduction
// result is needed alongside the elementwise operation, while ensuring the correct multiplication result.

__global__ void fusedMultiplyAndReduce(const float* __restrict__ A,
                                         float* __restrict__ C,
                                         float s,
                                         int64_t size,
                                         float* global_sum) {
    float local_sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop: each thread processes multiple elements
    for (int i = idx; i < size; i += stride) {
        float val = A[i] * s;
        C[i] = val;
        local_sum += val;
    }

    // Intra-warp reduction using warp shuffle primitives
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    // Each warp's leading thread writes its reduced sum to shared memory
    __shared__ float warpSums[32]; // supports up to 32 warps per block
    if (lane == 0) {
        warpSums[warpId] = local_sum;
    }
    __syncthreads();

    // Final reduction: first warp reduces the per-warp sums
    if (threadIdx.x < (blockDim.x / warpSize)) {
        float block_sum = warpSums[lane];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(global_sum, block_sum);
        }
    }
}

// The forward function performs fused matrix-scalar multiplication along with a reduction.
// It returns a vector of two tensors: the multiplied output and a 1-element tensor containing
// the sum of all elements in the output.

std::vector<torch::Tensor> forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    auto sum_tensor = torch::zeros({1}, A.options());
    int64_t size = A.numel();
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    fusedMultiplyAndReduce<<<blocks, threads>>>(A.data_ptr<float>(),
                                                  C.data_ptr<float>(),
                                                  s,
                                                  size,
                                                  sum_tensor.data_ptr<float>());

    return {C, sum_tensor};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused matrix-scalar multiplication with reduction kernel");
}
