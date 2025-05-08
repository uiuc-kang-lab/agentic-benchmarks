// fused_hinge_vec_reduction.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define a vectorized float4 structure with 16-byte alignment for optimal loads
struct __align__(16) Float4 {
    float x, y, z, w;
};

// Fused kernel that combines vectorized loads for coalesced memory access with an in-kernel reduction
// to compute the mean hinge loss: fmaxf(0, 1 - pred * targ).
// It processes the bulk of the data 4 elements at a time using Float4 and then handles any remainder
// elements separately. Reduction is performed using warp shuffle and shared memory, with per-block
// atomic additions to a global accumulator.

__global__ void fused_hinge_loss_vec_reduction_kernel(const float* __restrict__ predictions,
                                                       const float* __restrict__ targets,
                                                       float* global_sum,
                                                       int n) {
    float local_sum = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    // Process vectorized portion: groups of 4 elements
    int n4 = n / 4;  // Number of complete float4 groups
    const Float4* pred4 = reinterpret_cast<const Float4*>(predictions);
    const Float4* targ4 = reinterpret_cast<const Float4*>(targets);
    for (int i = tid; i < n4; i += gridSize) {
        Float4 p = __ldg(&pred4[i]);
        Float4 t = __ldg(&targ4[i]);
        float a = fmaxf(0.0f, 1.0f - p.x * t.x);
        float b = fmaxf(0.0f, 1.0f - p.y * t.y);
        float c = fmaxf(0.0f, 1.0f - p.z * t.z);
        float d = fmaxf(0.0f, 1.0f - p.w * t.w);
        local_sum += (a + b + c + d);
    }

    // Process any remaining elements
    int rem_start = n4 * 4;
    for (int i = rem_start + tid; i < n; i += gridSize) {
        float p = __ldg(&predictions[i]);
        float t = __ldg(&targets[i]);
        local_sum += fmaxf(0.0f, 1.0f - p * t);
    }

    // Intra-warp reduction using warp shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Allocate shared memory for warp-level partial sums
    __shared__ float shared[32];  // Enough for up to 1024 threads per block (32 warps)
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        shared[warpId] = local_sum;
    }
    __syncthreads();

    // Let the first warp reduce the per-warp results
    if (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) {
        local_sum = shared[lane];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
    }

    // The first thread in the block atomically adds its block's sum to the global sum
    if (threadIdx.x == 0) {
        atomicAdd(global_sum, local_sum);
    }
}

// The forward function validates the inputs, selects an appropriate block size based on n,
// launches the fused kernel, and computes the mean hinge loss by dividing the global sum by n.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();

    // Allocate a single-element tensor to accumulate the hinge loss sum.
    auto global_sum = torch::zeros({1}, predictions.options());

    // Dynamically choose a block size based on problem size
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

    fused_hinge_loss_vec_reduction_kernel<<<blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        global_sum.data_ptr<float>(),
        n
    );

    // Compute and return the mean hinge loss (performed on GPU)
    return global_sum / n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused Hinge Loss Forward Kernel with Vectorized Loads and Warp Shuffle Reduction");
}
