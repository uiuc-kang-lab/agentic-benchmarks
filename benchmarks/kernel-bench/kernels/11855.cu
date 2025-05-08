#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel distributes the workload evenly by using a computed chunk size for each thread,
// ensuring that every thread processes a contiguous block of elements, thereby avoiding bottlenecks.
__global__ void even_distribution_kldiv_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    // Compute equal workload per thread with ceiling division
    int chunk = (n + total_threads - 1) / total_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > n) end = n;

    float sum = 0.0f;
    // Each thread processes its assigned contiguous range
    for (int i = start; i < end; i++) {
        float lp = log_predictions[i];
        float tgt = targets[i];
        sum += expf(lp) - tgt * lp;
    }

    // Warp-level reduction using shuffle
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Write per-warp results to shared memory for block-level reduction
    __shared__ float shared_mem[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        shared_mem[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction across warps in the block
    if (threadIdx.x < 32) {
        float block_sum = (threadIdx.x < (blockDim.x + 31) / 32 ? shared_mem[threadIdx.x] : 0.0f);
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// CUDA function exposed to PyTorch
torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Define launch configuration (256 threads per block)
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    even_distribution_kldiv_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Even workload distribution KL divergence forward (CUDA)");
}
