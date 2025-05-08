#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Fused kernel: applies ELU activation and computes block-wise reduction sum using shared memory and warp-level primitives
__global__ void fused_elu_reduce(const float* __restrict__ x, float* __restrict__ out, 
                                   float alpha, float* __restrict__ block_sums, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? x[idx] : 0.f;
    float activated = (idx < n) ? ((val > 0.f) ? val : alpha * (expf(val) - 1.f)) : 0.f;
    if (idx < n) {
        out[idx] = activated;
    }

    // Each thread starts with its activated value
    float sum = activated;

    // Intra-warp reduction using warp shuffle primitives
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Write the reduced value of each warp into shared memory
    extern __shared__ float sdata[]; // Size = number of warps per block
    int lane = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        sdata[warpId] = sum;
    }
    __syncthreads();

    // Let the first warp reduce the warp sums
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < numWarps) {
        float blockSum = sdata[lane];
        for (int offset = numWarps / 2; offset > 0; offset /= 2) {
            blockSum += __shfl_down_sync(0xffffffff, blockSum, offset);
        }
        if (lane == 0) {
            block_sums[blockIdx.x] = blockSum;
        }
    }
}

// Final reduction kernel to sum the block sums into a single scalar
__global__ void final_reduce(const float* __restrict__ block_sums, float* __restrict__ out_sum, int num_blocks) {
    int tid = threadIdx.x;
    float sum = (tid < num_blocks) ? block_sums[tid] : 0.f;

    // Intra-warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float sdata[32]; // Enough for up to 256 threads (8 warps max)
    int lane = tid & (warpSize - 1);
    int warpId = tid / warpSize;
    if (lane == 0) {
        sdata[warpId] = sum;
    }
    __syncthreads();

    int warpCount = (blockDim.x + warpSize - 1) / warpSize;
    if (tid < warpCount) {
        float blockSum = sdata[lane];
        for (int offset = warpCount / 2; offset > 0; offset /= 2) {
            blockSum += __shfl_down_sync(0xffffffff, blockSum, offset);
        }
        if (tid == 0) {
            *out_sum = blockSum;
        }
    }
}

// The fused function applies ELU activation and computes the reduction sum of activated values.
// It returns a vector: [activated_tensor, reduction_sum_scalar].
torch::Tensor elu_cuda_fused(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    int n = x.numel();

    auto out = torch::empty_like(x);
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // Temporary tensor to hold block-level sums
    auto block_sums = torch::empty({blocks}, x.options());

    // Calculate shared memory size: one float per warp in each block
    int warps_per_block = (threads + 31) / 32;
    size_t shared_mem_size = warps_per_block * sizeof(float);

    fused_elu_reduce<<<blocks, threads, shared_mem_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, block_sums.data_ptr<float>(), n);

    auto out_sum = torch::empty({1}, x.options());
    if (blocks > 1) {
        // Launch final reduction kernel with 256 threads
        final_reduce<<<1, 256>>>(block_sums.data_ptr<float>(), out_sum.data_ptr<float>(), blocks);
    } else {
        // If only one block, copy its sum directly
        cudaMemcpy(out_sum.data_ptr<float>(), block_sums.data_ptr<float>(), sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Return both the elementwise activated tensor and the reduction sum as a scalar
    return {out, out_sum};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_fused, "Fused ELU activation with reduction (CUDA)");
}
