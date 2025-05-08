#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void cross_entropy_loss_kernel_shared(
    const float* logits,
    const int64_t* targets,
    float* losses,
    int batch_size,
    int num_classes
)
{
    extern __shared__ float shared_logits[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Process one sample per thread block
    if (blockIdx.x < batch_size) {
        // Load logits for this sample into shared memory
        for (int j = tid; j < num_classes; j += blockDim.x) {
            shared_logits[j] = logits[blockIdx.x * num_classes + j];
        }
        __syncthreads();

        // First pass: find maximum logit for numerical stability
        float max_logit = -INFINITY;
        for (int j = tid; j < num_classes; j += blockDim.x) {
            max_logit = fmaxf(max_logit, shared_logits[j]);
        }

        // Reduce max_logit within block
        for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                max_logit = fmaxf(max_logit, __shfl_down_sync(0xffffffff, max_logit, offset));
            }
        }
        max_logit = __shfl_sync(0xffffffff, max_logit, 0); // Broadcast max to all threads
        
        // Second pass: compute sum of exp(logits - max_logit)
        float sum_exp = 0.0f;
        for (int j = tid; j < num_classes; j += blockDim.x) {
            sum_exp += expf(shared_logits[j] - max_logit);
        }

        // Reduce sum_exp within block
        for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
            }
        }
        sum_exp = __shfl_sync(0xffffffff, sum_exp, 0); // Broadcast sum to all threads
        
        // Only thread 0 computes final loss
        if (tid == 0) {
            int target = targets[blockIdx.x];
            float log_sum_exp = logf(sum_exp);
            float target_logit = shared_logits[target];
            losses[blockIdx.x] = -(target_logit - max_logit - log_sum_exp);
        }
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

    // Allocate space for individual losses
    auto losses = torch::empty({batch_size}, predictions.options());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    size_t shared_memory_size = threads * sizeof(float);

    // Launch kernel using shared memory and grid-stride loop
    cross_entropy_loss_kernel_shared<<<blocks, threads, shared_memory_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel_shared: ", cudaGetErrorString(err));

    // Compute mean loss over the batch
    auto loss = losses.mean();
    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward with shared memory (CUDA)");
}