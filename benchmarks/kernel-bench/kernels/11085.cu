#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cfloat>

// Each block processes one sample. Threads within the block collaborate to compute the max
// and the sum of exponentials using shared memory. __syncthreads() is only used for shared
// memory consistency during the reduction phases.

__global__ void cross_entropy_loss_kernel_parallel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    int sample = blockIdx.x;  // each block processes one sample
    if (sample >= batch_size) return;

    const float* sample_logits = logits + sample * num_classes;
    
    // Declare shared memory for reduction (dynamically allocated)
    extern __shared__ float sdata[];

    // Step 1: Compute maximum logit for numerical stability
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < num_classes; j += blockDim.x) {
        float val = sample_logits[j];
        local_max = fmaxf(local_max, val);
    }

    // Store local max in shared memory
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    // Parallel reduction to get the maximum value
    for (int s = blockDim.x / 2; s > 32; s /= 2) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    // Warp-level reduction for final steps
    if (threadIdx.x < 32) {
        // Volatile pointer to prevent caching in registers
        volatile float* smem = sdata;
        if (blockDim.x >= 64) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + 32]);
        if (blockDim.x >= 32) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + 16]);
        if (blockDim.x >= 16) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + 8]);
        if (blockDim.x >= 8) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + 4]);
        if (blockDim.x >= 4) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + 2]);
        if (blockDim.x >= 2) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + 1]);
    }
    float max_val = sdata[0];

    // Step 2: Compute the sum of exp(logits - max_val)
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < num_classes; j += blockDim.x) {
        local_sum += expf(sample_logits[j] - max_val);
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Parallel reduction to sum up the exponentials
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float sum_exp = sdata[0];

    // Step 3: Compute and store the cross entropy loss for this sample
    if (threadIdx.x == 0) {
        int target = targets[sample];
        float logit_target = sample_logits[target];
        float loss = -(logit_target - max_val - logf(sum_exp));
        losses[sample] = loss;
    }
}


torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == batch_size, "targets must have same batch size as predictions");

    // Allocate output tensor for losses
    auto losses = torch::empty({batch_size}, predictions.options());

    // Use one block per sample; choose a fixed number of threads per block.
    int threads = 256;
    int blocks = batch_size;
    size_t shared_mem = threads * sizeof(float);

    cross_entropy_loss_kernel_parallel<<<blocks, threads, shared_mem>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel_parallel: ", cudaGetErrorString(err));

    // Compute mean loss over batch
    auto loss = losses.mean();
    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward with per-sample block parallelism (CUDA)");
}
