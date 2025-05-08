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
    int stride = blockDim.x * gridDim.x;

    // Load data into shared memory to reduce global memory accesses
    for (int offset = 0; offset < num_classes; offset += blockDim.x) {
        if (offset + tid < num_classes) {
            shared_logits[tid] = logits[idx * num_classes + offset + tid];
        }
        __syncthreads();
        
        // Process each sample with grid-stride loop and shared memory
        for (int i = idx; i < batch_size; i += stride) {
            float* logits_i = shared_logits;
            int target = targets[i];

            // Compute max for numerical stability
            float max_logit = logits_i[0];
            for (int j = 1; j < num_classes; j++) {
                max_logit = fmaxf(max_logit, logits_i[j]);
            }

            // Compute sum of exp(logits - max_logit)
            float sum_exp = 0.0f;
            if (tid < num_classes) { // Ensure we only loop within class count
                float target_shifted = logits_i[target] - max_logit;  // Cache this value
                for (int j = 0; j < num_classes; j++) {
                    float shifted_logit = logits_i[j] - max_logit;
                    sum_exp += expf(shifted_logit);
                }
                float log_sum_exp = logf(sum_exp);

                // Compute the cross entropy loss for the sample using cached value
                if (offset + tid == 0) {
                    losses[i] = -(target_shifted - log_sum_exp);
                }
            }

            __syncthreads(); // Ensure all threads have finished using shared memory
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