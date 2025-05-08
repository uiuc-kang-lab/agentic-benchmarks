#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// First kernel: compute per-block partial sum of hinge losses
__global__ void hinge_loss_reduction_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ block_sums,
    const int n) {

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;

    // Grid-stride loop: compute local sum for hinge loss
    for (int i = global_idx; i < n; i += stride) {
        float hinge = fmaxf(0.0f, 1.0f - predictions[i] * targets[i]);
        sum += hinge;
    }

    // Intra-warp reduction using warp-level primitives
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Shared memory for block-level reduction
    extern __shared__ float sdata[]; // size = (blockDim.x/warpSize) elements
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        sdata[warpId] = sum;
    }
    __syncthreads();

    // Let the first warp reduce the per-warp sums
    if (threadIdx.x < blockDim.x / warpSize) {
        float block_sum = sdata[lane];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        if (lane == 0) {
            block_sums[blockIdx.x] = block_sum;
        }
    }
}

// Second kernel: reduce the block sums to a single value
__global__ void final_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n) {
    int tid = threadIdx.x;
    int stride = blockDim.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        sum += input[i];
    }
    
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    
    __shared__ float sdata[32]; // assuming blockDim.x <= 256 so at most 8 warps
    int lane = tid % warpSize;
    int warpId = tid / warpSize;
    if(lane == 0) {
        sdata[warpId] = sum;
    }
    __syncthreads();
    
    int numWarps = blockDim.x / warpSize;
    if (tid < numWarps) {
        float final_sum = sdata[lane];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            final_sum += __shfl_down_sync(mask, final_sum, offset);
        }
        if (lane == 0) {
            output[0] = final_sum;
        }
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    const int n = predictions.numel();

    // Configure reduction kernel launch parameters
    const int threads = 256;
    const int max_blocks = 65535;
    const int blocks = std::min((n + threads - 1) / threads, max_blocks);

    // Allocate tensor for block-wise partial sums
    auto blockSums = torch::empty({blocks}, predictions.options());

    // Shared memory size: one float per warp in the block
    int sharedMemSize = (threads / warpSize) * sizeof(float);
    hinge_loss_reduction_kernel<<<blocks, threads, sharedMemSize>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        blockSums.data_ptr<float>(),
        n
    );

    float totalSum = 0.0f;
    if (blocks > 1) {
        // Final reduction: reduce blockSums to a single value
        auto finalSumTensor = torch::empty({1}, predictions.options());
        final_reduction_kernel<<<1, threads>>>(
            blockSums.data_ptr<float>(),
            finalSumTensor.data_ptr<float>(),
            blocks
        );
        cudaDeviceSynchronize();
        totalSum = finalSumTensor.item<float>();
    } else {
        cudaDeviceSynchronize();
        totalSum = blockSums.item<float>();
    }

    // Compute mean hinge loss
    float mean = totalSum / n;
    return torch::tensor(mean, predictions.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared Memory Reduction Hinge Loss Forward");
}
