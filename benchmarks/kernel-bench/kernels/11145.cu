#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// This kernel assigns one block per sample. Each thread in the block processes a portion of the num_classes dimension,
// then the block cooperatively performs two parallel reductions in shared memory: one to compute the maximum logit, and one to compute the sum of exponentials.

__global__ void cross_entropy_loss_kernel_2d(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) return;

    int tid = threadIdx.x;
    int blockSize = blockDim.x; // number of threads per block

    // Allocate shared memory for reduction: first half for max, second half for sum
    extern __shared__ float sdata[];  // size: 2 * blockSize * sizeof(float)
    float* smax = sdata;
    float* ssum = sdata + blockSize;

    // Pointer to the logits for this sample
    const float* sample_logits = logits + sample_idx * num_classes;

    // Step 1: Each thread computes a partial maximum over its portion of the classes
    float local_max = -FLT_MAX;
    for (int j = tid; j < num_classes; j += blockSize) {
        float val = sample_logits[j];
        if (val > local_max) local_max = val;
    }
    smax[tid] = local_max;
    __syncthreads();

    // Parallel reduction for maximum
    for (int stride = blockSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        }
        __syncthreads();
    }
    float global_max = smax[0];
    
    // Step 2: Each thread computes a partial sum of exp(logit - global_max) over its chunk
    float local_sum = 0.0f;
    for (int j = tid; j < num_classes; j += blockSize) {
        local_sum += expf(sample_logits[j] - global_max);
    }
    ssum[tid] = local_sum;
    __syncthreads();

    // Parallel reduction for sum
    for (int stride = blockSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }
    float global_sum = ssum[0];
    
    // Only thread 0 computes and writes the final loss for this sample
    if (tid == 0) {
        int target = targets[sample_idx];
        float loss = -(sample_logits[target] - global_max - logf(global_sum));
        losses[sample_idx] = loss;
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

    auto losses = torch::empty({batch_size}, predictions.options());

    // Launch one block per sample, with a fixed thread count per block.
    const int threads = 256;  // thread count per block
    const int blocks = batch_size;  // one block per sample

    // Shared memory size: two arrays of floats of length 'threads'
    size_t shared_mem_size = 2 * threads * sizeof(float);

    cross_entropy_loss_kernel_2d<<<blocks, threads, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in cross_entropy_loss_kernel_2d: ", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward with 2D grid reduction (CUDA)");
}
