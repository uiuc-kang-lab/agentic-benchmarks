#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// Optimized kernel using warp shuffle reductions and minimal shared memory
__global__ void cross_entropy_loss_kernel_opt(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int num_classes
) {
    // Each block processes one sample
    int sample = blockIdx.x;
    int tid = threadIdx.x;
    int offset = sample * num_classes;

    // Phase 1: Compute the maximum logit (for numerical stability) in parallel
    float local_max = -FLT_MAX;
    for (int j = tid; j < num_classes; j += blockDim.x) {
        float val = logits[offset + j];
        local_max = fmaxf(local_max, val);
    }

    // Intra-warp reduction for maximum using warp shuffle
    unsigned int mask = 0xffffffff;
    for (int off = warpSize / 2; off > 0; off /= 2) {
        float temp = __shfl_down_sync(mask, local_max, off);
        local_max = fmaxf(local_max, temp);
    }

    // Allocate dynamic shared memory for storing per-warp results
    // We only need one value per warp for max and sum
    extern __shared__ float sdata[];
    // Partition shared memory into two arrays: one for max and one for sum
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    float* s_max = sdata;           // size: numWarps floats
    float* s_sum = sdata + numWarps;  // size: numWarps floats

    int warpId = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) {
        s_max[warpId] = local_max;
    }
    __syncthreads();

    float max_val;
    // Let thread 0 perform reduction over the per-warp maximums
    if (tid == 0) {
        max_val = s_max[0];
        for (int i = 1; i < numWarps; i++) {
            max_val = fmaxf(max_val, s_max[i]);
        }
        s_max[0] = max_val; // Broadcast the global max
    }
    __syncthreads();
    max_val = s_max[0];

    // Phase 2: Compute the sum of exponentials using the computed max_val
    float local_sum = 0.0f;
    for (int j = tid; j < num_classes; j += blockDim.x) {
        float val = logits[offset + j];
        local_sum += expf(val - max_val);
    }
    
    // Intra-warp reduction for sum using warp shuffle
    for (int off = warpSize / 2; off > 0; off /= 2) {
        float temp = __shfl_down_sync(mask, local_sum, off);
        local_sum += temp;
    }

    if (lane == 0) {
        s_sum[warpId] = local_sum;
    }
    __syncthreads();

    float sum_exp;
    // Thread 0 reduces the per-warp sums
    if (tid == 0) {
        sum_exp = 0.0f;
        for (int i = 0; i < numWarps; i++) {
            sum_exp += s_sum[i];
        }
        s_sum[0] = sum_exp;
    }
    __syncthreads();
    sum_exp = s_sum[0];

    // Final loss computation performed by thread 0
    if (tid == 0) {
        int target = targets[sample];
        float target_logit = logits[offset + target];
        float loss = -(target_logit - max_val - logf(sum_exp));
        losses[sample] = loss;
    }
}


// Host function interfacing with PyTorch
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be a Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be an Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == batch_size, "targets must have the same batch size as predictions");

    // Create output tensor for losses
    auto losses = torch::empty({batch_size}, predictions.options());

    // Launch one block per sample; choose an appropriate number of threads per block
    int threads = 128; // Tuning parameter; should be a multiple of warpSize
    int blocks = batch_size;
    // Allocate dynamic shared memory only for per-warp reductions: two arrays of size numWarps each
    int numWarps = (threads + 31) / 32;
    size_t shared_mem_size = 2 * numWarps * sizeof(float);

    cross_entropy_loss_kernel_opt<<<blocks, threads, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel_opt: ", cudaGetErrorString(err));

    // Compute and return the mean loss over the batch
    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Cross Entropy Loss forward (CUDA) with warp-level reductions");
}
