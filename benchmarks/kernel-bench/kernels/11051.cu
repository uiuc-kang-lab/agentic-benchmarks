#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

// Kernel that computes cross entropy loss per sample using shared memory
// for reductions and warp-level primitives for intra-warp communication.

__global__ void cross_entropy_loss_kernel_shared(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int num_samples,
    int num_classes) {

    // Each block handles one sample
    int sample_id = blockIdx.x;
    if (sample_id >= num_samples) return;

    // Pointer to the logits for this sample
    const float* sample_logits = logits + sample_id * num_classes;

    // Read the target class for this sample
    int target = targets[sample_id];

    // Each block will use its threads to reduce over num_classes
    // First reduction: compute the maximum logit for numerical stability
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < num_classes; j += blockDim.x) {
        float val = sample_logits[j];
        local_max = fmaxf(local_max, val);
    }

    // Intra-warp reduction for local max using warp-level primitives
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(mask, local_max, offset);
        local_max = fmaxf(local_max, other);
    }

    // Allocate shared memory for warp-level results
    extern __shared__ float sdata[]; // size = (blockDim.x/warpSize) * sizeof(float)
    int warp_id = threadIdx.x / warpSize;
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        sdata[warp_id] = local_max;
    }
    __syncthreads();

    float global_max = local_max;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < numWarps) {
        float temp = sdata[threadIdx.x];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            temp = fmaxf(temp, __shfl_down_sync(mask, temp, offset));
        }
        if (threadIdx.x == 0) {
            global_max = temp;
            sdata[0] = global_max; // store global max for broadcast
        }
    }
    __syncthreads();
    global_max = sdata[0];

    // Second reduction: compute sum of exp(logit - global_max)
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < num_classes; j += blockDim.x) {
        local_sum += expf(sample_logits[j] - global_max);
    }

    // Intra-warp reduction for local sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(mask, local_sum, offset);
        local_sum += other;
    }

    if ((threadIdx.x & (warpSize - 1)) == 0) {
        sdata[warp_id] = local_sum;
    }
    __syncthreads();

    float global_sum = local_sum;
    if (threadIdx.x < numWarps) {
        float temp = sdata[threadIdx.x];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            temp += __shfl_down_sync(mask, temp, offset);
        }
        if (threadIdx.x == 0) {
            global_sum = temp;
            sdata[0] = global_sum;
        }
    }
    __syncthreads();
    global_sum = sdata[0];

    // Use thread 0 to compute the final loss for this sample
    if (threadIdx.x == 0) {
        float true_logit = sample_logits[target];
        float loss = -(true_logit - global_max - logf(global_sum));
        losses[sample_id] = loss;
    }
}


// Host function: wraps the kernel call

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be Int64 tensor");

    int num_samples = predictions.size(0);
    int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == num_samples, "targets must have same batch size as predictions");

    // Allocate output tensor for losses
    auto losses = torch::empty({num_samples}, predictions.options());

    // Launch one block per sample with a chosen number of threads
    int threads = 128; // number of threads per block
    // Shared memory size: one float per warp
    size_t shared_mem_size = ((threads + warpSize - 1) / warpSize) * sizeof(float);
    dim3 blocks(num_samples);

    cross_entropy_loss_kernel_shared<<<blocks, threads, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        num_samples,
        num_classes);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel_shared: ", cudaGetErrorString(err));

    // Compute mean loss over the batch
    auto loss = losses.mean();
    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward (CUDA)");
}
