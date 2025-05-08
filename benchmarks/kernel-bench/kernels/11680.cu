#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that computes the KL divergence using reduction in shared memory
// and dynamically chosen block size for optimal performance
__global__ void block_size_tuning_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    // Each thread processes multiple elements using a stride pattern
    for (int i = idx; i < n; i += stride) {
        float log_val = log_predictions[i];
        float tgt = targets[i];
        sum += expf(log_val) - tgt * log_val;
    }

    // Store partial sum in shared memory
    shared[tid] = sum;
    __syncthreads();

    // Reduction within block in shared memory
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // Atomic accumulation of each block's contribution to global output
    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

// Host function that dynamically chooses block size from candidate values
// Based on the problem size, we pick one of: 32, 64, 128, 256, 512
// and launch the kernel accordingly. This helps in reducing runtime by
// matching the workload with the optimal GPU occupancy configuration.

torch::Tensor block_size_tuning_kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    int block_size;
    if (n < 2048) {
        block_size = 32;
    } else if (n < 8192) {
        block_size = 64;
    } else if (n < 32768) {
        block_size = 128;
    } else if (n < 131072) {
        block_size = 256;
    } else {
        block_size = 512;
    }

    // Calculate grid size and limit to a reasonable maximum
    int grid = (n + block_size - 1) / block_size;
    grid = min(grid, 256);

    size_t shared_mem = block_size * sizeof(float);

    block_size_tuning_kl_div_kernel<<<grid, block_size, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    cudaDeviceSynchronize();
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &block_size_tuning_kl_div_cuda_forward, "KL divergence forward with dynamic block size tuning (CUDA)");
}
