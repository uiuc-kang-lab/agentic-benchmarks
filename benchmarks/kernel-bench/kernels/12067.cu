#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define a 16-byte aligned structure to load 4 floats at once
struct __align__(16) Float4 {
    float x, y, z, w;
};

// This fused kernel uses vectorized loads (via Float4) for the bulk of the data and
// a grid-stride loop combined with warp shuffle based reduction to compute the hinge loss sum.
// It also processes any leftover elements that don't fit into a set of 4.
__global__ void fused_vectorized_hinge_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* global_sum,
    int n) {

    float local_sum = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    // Process aligned elements in groups of 4
    int n4 = n / 4;
    const Float4* predictions4 = reinterpret_cast<const Float4*>(predictions);
    const Float4* targets4 = reinterpret_cast<const Float4*>(targets);

    for (int i = tid; i < n4; i += gridSize) {
        Float4 p = __ldg(&predictions4[i]);
        Float4 t = __ldg(&targets4[i]);
        local_sum += fmaxf(0.0f, 1.0f - p.x * t.x);
        local_sum += fmaxf(0.0f, 1.0f - p.y * t.y);
        local_sum += fmaxf(0.0f, 1.0f - p.z * t.z);
        local_sum += fmaxf(0.0f, 1.0f - p.w * t.w);
    }

    // Process any remaining elements that don't fit in a group of 4
    int remainder_start = n4 * 4;
    for (int i = remainder_start + tid; i < n; i += gridSize) {
        float p = __ldg(&predictions[i]);
        float t = __ldg(&targets[i]);
        local_sum += fmaxf(0.0f, 1.0f - p * t);
    }

    // Intra-warp reduction using warp shuffle instructions
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Shared memory to accumulate sums from each warp within the block
    __shared__ float shared[32];  // Supports up to 1024 threads per block (32 warps)
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        shared[warpId] = local_sum;
    }
    __syncthreads();

    // Let the first warp finish reducing the per-warp sums
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    local_sum = (threadIdx.x < numWarps) ? shared[lane] : 0.0f;
    if (warpId == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
    }

    // The first thread in the block atomically adds the block result to the global sum
    if (threadIdx.x == 0) {
        atomicAdd(global_sum, local_sum);
    }
}

// The forward function validates inputs, selects an appropriate block size, launches the fused kernel,
// and finally computes the mean hinge loss by dividing the total loss summed over all elements by n.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();

    // Allocate a tensor to hold the global loss sum
    auto global_sum = torch::zeros({1}, predictions.options());

    // Dynamic block size selection based on input size
    int block_size;
    if (n < 512) {
        block_size = 32;
    } else if (n < 4096) {
        block_size = 64;
    } else if (n < 100000) {
        block_size = 128;
    } else if (n < 10000000) {
        block_size = 256;
    } else {
        block_size = 512;
    }
    int blocks = (n + block_size - 1) / block_size;

    fused_vectorized_hinge_loss_kernel<<<blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        global_sum.data_ptr<float>(),
        n
    );

    // Compute the mean hinge loss using GPU arithmetic
    return global_sum / n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused Vectorized Hinge Loss with In-Kernel Reduction");
}
