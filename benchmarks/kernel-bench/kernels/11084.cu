#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// Each block handles one sample, and threads in the block cooperatively compute the maximum and the sum of exponentials using shared memory reductions.
__global__ void cross_entropy_loss_kernel_parallel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int num_classes
) {
    // Each block processes one sample.
    int i = blockIdx.x;  // sample index
    int tid = threadIdx.x;

    // Allocate shared memory for reductions, split into two arrays.
    extern __shared__ float sdata[];
    float* s_max = sdata;               // For max reduction
    float* s_sum = sdata + blockDim.x;    // For sum reduction

    // Phase 1: Compute the maximum logit for numerical stability
    float local_max = -FLT_MAX;
    for (int j = tid; j < num_classes; j += blockDim.x) {
        float val = logits[i * num_classes + j];
        local_max = fmaxf(local_max, val);
    }
    s_max[tid] = local_max;
    __syncthreads();

    // Parallel reduction for maximum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    float max_val = s_max[0];
    
    // Phase 2: Compute the sum of exp(logit - max) in parallel
    float local_sum = 0.0f;
    for (int j = tid; j < num_classes; j += blockDim.x) {
        local_sum += expf(logits[i * num_classes + j] - max_val);
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    // Parallel reduction for the sum of exponentials
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 computes the final loss for the sample
    if (tid == 0) {
        int target = targets[i];
        float target_logit = logits[i * num_classes + target];
        float loss = -(target_logit - max_val - logf(s_sum[0]));
        losses[i] = loss;
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

    // Launch one block per sample, with 128 threads per block for better occupancy
    int threads = 128;  // Reduced from 256 to 128 for potentially better occupancy
    int blocks = batch_size;
    size_t shared_mem_size = 2 * threads * sizeof(float); // Shared memory for max and sum reductions

    cross_entropy_loss_kernel_parallel<<<blocks, threads, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel_parallel: ", cudaGetErrorString(err));

    // Compute and return the mean loss over the batch
    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Parallel Cross Entropy Loss forward (CUDA)");
}
