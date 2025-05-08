// fused_hinge_loss.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This fused kernel computes the hinge loss ( max(0, 1 - pred * targ) ) and reduces the result to a global sum.
// It uses a grid-stride loop to process arbitrary sized input and utilizes warp shuffle instructions
// for efficient in-warp reductions, followed by a shared memory reduction across warps. Finally, block-level
// results are atomically added to a global result, avoiding a separate reduction kernel call.

__global__ void fused_hinge_loss_kernel(const float* __restrict__ predictions,
                                          const float* __restrict__ targets,
                                          float* global_sum,
                                          int n) {
    float local_sum = 0.0f;
    // Each thread processes multiple elements with a grid-stride loop
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += gridSize) {
        float pred = __ldg(&predictions[i]);
        float targ = __ldg(&targets[i]);
        local_sum += fmaxf(0.0f, 1.0f - pred * targ);
    }

    // Intra-warp reduction using warp shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Allocate shared memory for partial sums from each warp
    __shared__ float shared[32];  // Enough for up to 1024 threads per block (32 warps)
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        shared[warpId] = local_sum;
    }
    __syncthreads();

    // Let the first warp in the block reduce the per-warp results
    local_sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
    if (warpId == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
    }

    // The first thread of each block atomically accumulates the block's sum into global_sum
    if (threadIdx.x == 0) {
        atomicAdd(global_sum, local_sum);
    }
}

// The forward function validates inputs, selects an appropriate block size based on n, launches the fused kernel,
// and finally computes the mean hinge loss by dividing the accumulated sum by n.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();

    // Initialize the global sum on GPU
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

    fused_hinge_loss_kernel<<<blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        global_sum.data_ptr<float>(),
        n
    );

    // Compute mean hinge loss (performed on GPU)
    return global_sum / n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused Hinge Loss Forward Kernel with Warp Shuffle Reduction");
}
