#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function to compute the maximum logit for numerical stability
__device__ __forceinline__ float compute_max_logit(const float* logits, int num_classes) {
    float max_logit = logits[0];
    for (int i = 1; i < num_classes; i++) {
        max_logit = fmaxf(max_logit, logits[i]);
    }
    return max_logit;
}

// Device function to compute the sum of exponentials, subtracting the max logit
__device__ __forceinline__ float compute_sum_exp(const float* logits, int num_classes, float max_logit) {
    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        sum_exp += expf(logits[i] - max_logit);
    }
    return sum_exp;
}

// Optimized kernel for computing cross entropy loss
__global__ void cross_entropy_loss_kernel_optimized(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    extern __shared__ float shared_mem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Use shared memory to reduce global memory access latencies
    float* logits_sample_shared = shared_mem + threadIdx.x * num_classes;

    // Load logits into shared memory
    const float* logits_sample = logits + idx * num_classes;
    for (int j = 0; j < num_classes; ++j) {
        logits_sample_shared[j] = logits_sample[j];
    }
    __syncthreads(); // Ensure all threads have loaded the data

    int64_t target = targets[idx];
  
    // Use modular functions for computation
    float max_logit = compute_max_logit(logits_sample_shared, num_classes);
    float sum_exp = compute_sum_exp(logits_sample_shared, num_classes, max_logit);
    
    // Compute the final loss for the sample
    float loss = -(logits_sample_shared[target] - max_logit - logf(sum_exp));
    losses[idx] = loss;
}

// Forward function for the CUDA module
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

    // Allocate output tensor for losses per sample
    auto losses = torch::empty({batch_size}, predictions.options());

    // Configure kernel launch parameters
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    size_t shared_mem_size = threads * num_classes * sizeof(float);

    cross_entropy_loss_kernel_optimized<<<blocks, threads, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel_optimized: ", cudaGetErrorString(err));

    // Compute the mean loss over the batch
    auto loss = losses.mean();
    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward (CUDA)");
}