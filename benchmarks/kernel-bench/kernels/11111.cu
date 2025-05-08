#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void cross_entropy_loss_kernel_shared(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
)
{
    extern __shared__ float shared_logits[];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int tid = threadIdx.x;

    for (int i = idx; i < batch_size; i += stride) {
        const float* logits_i = logits + i * num_classes;
        const int target = targets[i];
        float max_logit = -INFINITY;

        // Load logits into shared memory and compute max simultaneously
        #pragma unroll 4
        for (int j = tid; j < num_classes; j += blockDim.x) {
            const float val = logits_i[j];
            shared_logits[j] = val;
            max_logit = fmaxf(max_logit, val);
        }
        __syncthreads();

        // Reduce max across threads in the block
        #pragma unroll 5
        for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                max_logit = fmaxf(max_logit, __shfl_down_sync(0xffffffff, max_logit, offset));
            }
        }
        
        // Broadcast max_logit to all threads
        max_logit = __shfl_sync(0xffffffff, max_logit, 0);

        // Compute sum of exp(logits - max_logit)
        float sum_exp = 0.0f;
        const float target_shifted = shared_logits[target] - max_logit;

        #pragma unroll 4
        for (int j = 0; j < num_classes; j++) {
            sum_exp += expf(shared_logits[j] - max_logit);
        }

        const float log_sum_exp = logf(sum_exp);
        losses[i] = -(target_shifted - log_sum_exp);
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
    int shared_mem_size = num_classes * sizeof(float);

    // Launch kernel using grid-stride loop for better load balancing
    cross_entropy_loss_kernel_shared<<<blocks, threads, shared_mem_size>>>(
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