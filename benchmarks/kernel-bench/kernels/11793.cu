#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel evenly divides the input range among a fixed number of threads to ensure balanced workload.
// Each thread computes a contiguous subrange of the input data and calculates its partial KL divergence sum.
// A block-level reduction is then performed using shared memory, and the final results are accumulated into a global output via atomicAdd.

__global__ void evenly_distributed_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int n,
    int numThreads) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numThreads) return;

    // Compute contiguous subrange boundaries for this thread
    int start = (tid * n) / numThreads;
    int end = ((tid + 1) * n) / numThreads;
    float local_sum = 0.0f;
    
    // Compute partial sum over the assigned contiguous subrange
    for (int i = start; i < end; i++) {
        float lp = log_predictions[i];
        local_sum += expf(lp) - targets[i] * lp;
    }

    // Use shared memory for block-level reduction
    extern __shared__ float sdata[];
    int tid_local = threadIdx.x;
    sdata[tid_local] = local_sum;
    __syncthreads();

    // Reduce the block's values in shared memory
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid_local < s) {
            sdata[tid_local] += sdata[tid_local + s];
        }
        __syncthreads();
    }

    // The first thread in the block adds the block's sum to the global output
    if (tid_local == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// Host function to launch the kernel. It calculates the total number of threads to use based on the problem size
// (capped at 65,536 for balanced distribution) to ensure even workload across threads and blocks.

torch::Tensor evenly_distributed_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Use up to 65,536 threads for balanced workload distribution
    int numThreads = (n < 65536 ? n : 65536);
    int threads = 256;
    int blocks = (numThreads + threads - 1) / threads;
    int shared_mem = threads * sizeof(float);

    evenly_distributed_kl_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        numThreads
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &evenly_distributed_kl_forward, "Evenly distributed KL divergence (CUDA)");
}
