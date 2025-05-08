#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// Define warp size
#define WARP_SIZE 32

// Warp-level reduction for maximum
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// Warp-level reduction for sum
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Combined kernel: uses a 2D thread block mapping (rows = samples, columns = reduction over classes)
// and employs warp-level primitives to do inter-warp reductions with minimal shared memory usage.
// Each sample is processed by a row of threads; each thread iterates over multiple class indices.
// First, the maximum logit is computed (for numerical stability), then the sum of exponentials is computed.
// Finally, thread 0 in each sample row computes the final cross entropy loss.

__global__ void cross_entropy_loss_kernel_efficient(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    // Map blockIdx.x and threadIdx.y to a sample index
    int sample = blockIdx.x * blockDim.y + threadIdx.y;
    if (sample >= batch_size) return;

    const float* sample_logits = logits + sample * num_classes;

    // Each sample is processed by a row of blockDim.x threads.
    // Assume blockDim.x is a multiple of WARP_SIZE.
    int warpsPerRow = blockDim.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;

    // Phase 1: Compute maximum logit locally
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < num_classes; j += blockDim.x) {
        float val = sample_logits[j];
        local_max = fmaxf(local_max, val);
    }
    // Intra-warp reduction
    local_max = warpReduceMax(local_max);

    // Use shared memory for inter-warp reduction per sample row
    // Shared memory layout:
    // First: an array for storing per-warp results for max (size: blockDim.y * warpsPerRow)
    // Second: an array for storing the final per-row value (size: blockDim.y) reused for both max and sum reductions.
    extern __shared__ float shared[];
    float* smax = shared;                         // size: blockDim.y * warpsPerRow
    float* sfinal = shared + blockDim.y * warpsPerRow; // size: blockDim.y

    // Each warp writes its reduced maximum for this sample row
    smax[threadIdx.y * warpsPerRow + warpId] = local_max;
    __syncthreads();

    float max_val;
    if (threadIdx.x == 0) {  // one thread per sample row performs the final reduction
        max_val = smax[threadIdx.y * warpsPerRow];
        for (int i = 1; i < warpsPerRow; i++) {
            max_val = fmaxf(max_val, smax[threadIdx.y * warpsPerRow + i]);
        }
        sfinal[threadIdx.y] = max_val;  // store final max for this row
    }
    __syncthreads();
    max_val = sfinal[threadIdx.y]; // all threads in the row load the final maximum

    // Phase 2: Compute sum of exponentials with numerical stability
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < num_classes; j += blockDim.x) {
        local_sum += expf(sample_logits[j] - max_val);
    }
    local_sum = warpReduceSum(local_sum);

    // Reuse smax for inter-warp reduction for sum (overwriting previous content)
    smax[threadIdx.y * warpsPerRow + warpId] = local_sum;
    __syncthreads();

    float sum_exp;
    if (threadIdx.x == 0) {
        sum_exp = smax[threadIdx.y * warpsPerRow];
        for (int i = 1; i < warpsPerRow; i++) {
            sum_exp += smax[threadIdx.y * warpsPerRow + i];
        }
        sfinal[threadIdx.y] = sum_exp; // store final sum for row
    }
    __syncthreads();
    sum_exp = sfinal[threadIdx.y];

    // Phase 3: Compute final cross entropy loss; only thread 0 does this for each sample row
    if (threadIdx.x == 0) {
        int64_t target = targets[sample];
        float logit_target = sample_logits[target];
        float loss = -(logit_target - max_val - logf(sum_exp));
        losses[sample] = loss;
    }
}

// Host wrapper
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == batch_size, "targets must have same batch size as predictions");

    auto losses = torch::empty({batch_size}, predictions.options());

    // Define 2D block: x-dimension for intra-sample reduction, y-dimension for processing multiple samples per block
    const int threads_x = 128; // must be a multiple of 32
    const int threads_y = 4;   
    dim3 block(threads_x, threads_y);
    int grid_x = (batch_size + threads_y - 1) / threads_y;
    dim3 grid(grid_x);

    // Calculate shared memory size:
    // For each block row (sample), we need warpsPerRow floats and one extra float for the final reduction.
    int warpsPerRow = threads_x / WARP_SIZE;
    size_t shared_mem_size = block.y * (warpsPerRow + 1) * sizeof(float);

    cross_entropy_loss_kernel_efficient<<<grid, block, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch error: ", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Efficient Cross Entropy Loss forward (CUDA) using warp-level reductions and 2D thread blocks");
}
