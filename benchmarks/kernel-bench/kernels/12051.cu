#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Warp-level reduction using shuffle intrinsics
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Hybrid kernel: grid-stride loop with warp shuffle reduction and shared memory reduction
// Templated on BLOCK_SIZE (must be a multiple of warpSize)

template <int BLOCK_SIZE>
__global__ void hinge_loss_hybrid_kernel(const float* __restrict__ predictions,
                                           const float* __restrict__ targets,
                                           float* __restrict__ partialSums,
                                           int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;
    float localSum = 0.0f;

    // Grid-stride loop to accumulate each thread's partial sum
    for (int i = idx; i < n; i += gridDim.x * BLOCK_SIZE) {
        float pred = __ldg(&predictions[i]);
        float targ = __ldg(&targets[i]);
        localSum += fmaxf(0.0f, 1.0f - pred * targ);
    }

    // Intra-warp reduction
    localSum = warp_reduce_sum(localSum);

    // Each warp leader writes its result into shared memory
    __shared__ float warpSums[BLOCK_SIZE / 32];
    int warpId = tid / warpSize;
    if (tid % warpSize == 0) {
        warpSums[warpId] = localSum;
    }
    __syncthreads();

    // First warp reduces the warp sums
    float blockSum = 0.0f;
    int numWarps = BLOCK_SIZE / 32;
    if (tid < numWarps) {
        blockSum = warpSums[tid];
        blockSum = warp_reduce_sum(blockSum);
        if (tid == 0) {
            partialSums[blockIdx.x] = blockSum;
        }
    }
}

// Forward function: selects optimal block size and launches the hybrid kernel

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();

    // Choose block size according to problem size
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

    // Compute the number of blocks needed
    int blocks = (n + block_size - 1) / block_size;
    auto partialSums = torch::empty({blocks}, predictions.options());

    // Launch the appropriate templated kernel
    switch(block_size) {
        case 32:
            hinge_loss_hybrid_kernel<32><<<blocks, 32>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                partialSums.data_ptr<float>(),
                n
            );
            break;
        case 64:
            hinge_loss_hybrid_kernel<64><<<blocks, 64>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                partialSums.data_ptr<float>(),
                n
            );
            break;
        case 128:
            hinge_loss_hybrid_kernel<128><<<blocks, 128>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                partialSums.data_ptr<float>(),
                n
            );
            break;
        case 256:
            hinge_loss_hybrid_kernel<256><<<blocks, 256>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                partialSums.data_ptr<float>(),
                n
            );
            break;
        case 512:
            hinge_loss_hybrid_kernel<512><<<blocks, 512>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                partialSums.data_ptr<float>(),
                n
            );
            break;
        default:
            hinge_loss_hybrid_kernel<256><<<blocks, 256>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                partialSums.data_ptr<float>(),
                n
            );
            break;
    }

    // Final reduction is performed on the GPU using torch::sum
    return torch::sum(partialSums) / n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid Hinge Loss Forward Kernel");
}
