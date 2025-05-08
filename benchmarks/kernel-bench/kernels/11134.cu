#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel that computes per-sample cross entropy loss and performs a block reduction in shared memory,
// using an atomicAdd only once per block to accumulate the global loss sum.

__global__ void cross_entropy_loss_reduce_kernel(
    const float* logits,
    const int64_t* targets,
    int batch_size,
    int num_classes,
    float* global_loss
) {
    extern __shared__ float sdata[];  // Shared memory for block-level reduction
    int tid = threadIdx.x;
    int globalThreadId = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    float local_sum = 0.0f;

    // Each thread processes multiple samples in a stride.
    for (int i = globalThreadId; i < batch_size; i += stride) {
        const float* logits_i = logits + i * num_classes;
        int64_t target = targets[i];

        // Compute maximum logit for numerical stability
        float max_logit = logits_i[0];
        for (int j = 1; j < num_classes; j++) {
            float val = logits_i[j];
            if (val > max_logit) {
                max_logit = val;
            }
        }

        // Compute sum of exponentials
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            sum_exp += expf(logits_i[j] - max_logit);
        }

        // Compute log(sum_exp) and loss for the sample
        float log_sum_exp = logf(sum_exp);
        float loss = - (logits_i[target] - max_logit - log_sum_exp);
    
        local_sum += loss;
    }

    // Each thread writes its local sum into shared memory
    sdata[tid] = local_sum;
    __syncthreads();

    // Perform block-level reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Only one atomic operation per block to add block sum to global accumulator
    if (tid == 0) {
        atomicAdd(global_loss, sdata[0]);
    }
}

// Forward function: launches the reduction kernel to compute the mean cross entropy loss

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be a Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be an Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == batch_size, "Batch size of predictions and targets must match");

    // Allocate a single-element tensor for the global loss sum and initialize it to zero
    auto opts = predictions.options();
    auto global_loss_tensor = torch::zeros({1}, opts);

    // Set up CUDA kernel launch parameters
    int threads = 512;
    int blocks = (batch_size + threads - 1) / threads;
    size_t sharedMemSize = threads * sizeof(float);

    cross_entropy_loss_reduce_kernel<<<blocks, threads, sharedMemSize>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        batch_size,
        num_classes,
        global_loss_tensor.data_ptr<float>()
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_reduce_kernel: ", cudaGetErrorString(err));

    // Compute mean loss by dividing global loss sum by the batch size
    auto mean_loss = global_loss_tensor / static_cast<float>(batch_size);
    return mean_loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward with atomic minimal reduction (CUDA)");
}
