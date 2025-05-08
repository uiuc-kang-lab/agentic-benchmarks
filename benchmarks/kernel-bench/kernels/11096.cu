#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void optimized_cross_entropy_loss_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int i = blockIdx.x;

    // Each block processes one sample
    if (i < batch_size) {
        const float* logits_i = logits + i * num_classes;
        int target = targets[i];

        // Compute max for numerical stability
        float local_max = -FLT_MAX;
        for (int j = tid; j < num_classes; j += blockDim.x) {
            local_max = fmaxf(local_max, logits_i[j]);
        }
        shared_data[tid] = local_max;
        __syncthreads();

        // Reduce to find the maximum logit
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + s]);
            }
            __syncthreads();
        }
        float max_logit = shared_data[0];

        // Compute sum of exp(logits - max_logit)
        float local_sum = 0.0f;
        for (int j = tid; j < num_classes; j += blockDim.x) {
            local_sum += expf(logits_i[j] - max_logit);
        }
        shared_data[tid] = local_sum;
        __syncthreads();

        // Reduce to find the sum of exponentials
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_data[tid] += shared_data[tid + s];
            }
            __syncthreads();
        }
        float sum_exp = shared_data[0];

        // Compute the cross entropy loss for the sample
        if (tid == 0) {
            float log_sum_exp = logf(sum_exp);
            losses[i] = -(logits_i[target] - max_logit - log_sum_exp);
        }
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be a Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be an Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == batch_size, "targets must have same batch size as predictions");

    // Output losses tensor
    auto losses = torch::empty({batch_size}, predictions.options());

    int threads = 128;
    int blocks = batch_size;
    size_t shared_mem_size = threads * sizeof(float); // Shared memory for reductions

    optimized_cross_entropy_loss_kernel<<<blocks, threads, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in optimized_cross_entropy_loss_kernel: ", cudaGetErrorString(err));

    // Compute and return the mean loss over the batch
    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Cross Entropy Loss forward (CUDA)");
}
