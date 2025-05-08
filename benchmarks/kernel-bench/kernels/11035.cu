#include <torch/extension.h>
#include <cmath>
#include <cfloat>

// Optimized CUDA kernel using parallel reduction in shared memory.
// Each block processes one sample from the batch.

__global__ void cross_entropy_loss_kernel_optimized(
    const float* logits,
    const int64_t* targets,
    float* losses,
    int num_classes
) {
    extern __shared__ float sdata[]; // shared memory for reductions
    int sample = blockIdx.x;  // each block handles one sample
    int tid = threadIdx.x;
    
    // Pointer to the logits for the current sample
    const float* logits_i = logits + sample * num_classes;
    
    // Phase 1: Compute maximum logit for numerical stability
    float local_max = -FLT_MAX;
    for (int j = tid; j < num_classes; j += blockDim.x) {
        float logit = logits_i[j];
        if (logit > local_max) {
            local_max = logit;
        }
    }

    // Store the local maximum in shared memory
    sdata[tid] = local_max;
    __syncthreads();

    // Reduction to obtain the maximum logit in the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float max_logit = sdata[0];

    // Phase 2: Compute the sum of exponentials in parallel
    float local_sum = 0.0f;
    for (int j = tid; j < num_classes; j += blockDim.x) {
        local_sum += expf(logits_i[j] - max_logit);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Reduction to sum all exponential terms in the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum_exp = sdata[0];

    // Thread 0 computes the final loss for the sample
    if (tid == 0) {
        int64_t target = targets[sample];
        float loss = - (logits_i[target] - max_logit - logf(sum_exp));
        losses[sample] = loss;
    }
}


torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    // Ensure inputs are valid and on CUDA
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be a Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be an Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == batch_size, "targets must have same batch size as predictions");

    // Allocate output tensor for per-sample losses
    auto losses = torch::empty({batch_size}, predictions.options());

    // Launch one block per sample using 256 threads per block
    int threads = 256;
    int blocks = batch_size;
    size_t shared_memory_size = threads * sizeof(float);

    cross_entropy_loss_kernel_optimized<<<blocks, threads, shared_memory_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        num_classes
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel_optimized: ", cudaGetErrorString(err));

    // Compute mean loss over the batch
    auto loss = losses.mean();
    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Cross Entropy Loss forward (CUDA)");
}
