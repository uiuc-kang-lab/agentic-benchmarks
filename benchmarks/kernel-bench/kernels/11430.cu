#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel leverages shared memory to load a tile of log_predictions and targets
// into shared memory, reducing global memory latency. Each block processes multiple
// tiles in a grid-stride loop. After computing the per-element KL divergence using
// the shared data, a block-level reduction is performed and the result is accumulated
// into global memory via atomics.

__global__ void kldiv_tiled_shared_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Allocate shared memory for a tile: first blockDim.x for log_predictions,
    // next blockDim.x for targets.
    extern __shared__ float sdata[]; // total size allocated: 2 * blockDim.x * sizeof(float)
    float* s_log = sdata;
    float* s_target = sdata + blockDim.x;

    float local_sum = 0.0f;
    const int tile_size = blockDim.x; // each tile has one element per thread

    // Loop over tiles with grid-stride to cover the entire input
    for (int base = blockIdx.x * tile_size; base < n; base += gridDim.x * tile_size) {
        int idx = base + threadIdx.x;

        // Load tile data from global memory into shared memory
        if (idx < n) {
            s_log[threadIdx.x] = log_predictions[idx];
            s_target[threadIdx.x] = targets[idx];
        } else {
            s_log[threadIdx.x] = 0.0f;
            s_target[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Compute KL divergence for the element in the tile
        if (idx < n) {
            float val = s_log[threadIdx.x];
            float tgt = s_target[threadIdx.x];
            local_sum += expf(val) - tgt * val;
        }
        __syncthreads();
    }

    // Reduction of local_sum across threads in the block
    // Reuse the first part of shared memory for reduction
    float* block_sums = sdata;
    block_sums[threadIdx.x] = local_sum;
    __syncthreads();

    // Parallel reduction within the block using binary tree reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_sums[threadIdx.x] += block_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Atomically add the block's sum to global output
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sums[0]);
    }
}


torch::Tensor kldiv_tiled_shared_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    // Allocate shared memory: two arrays of 'threads' floats each
    int shared_mem = 2 * threads * sizeof(float);

    kldiv_tiled_shared_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kldiv_tiled_shared_cuda_forward, "Tiled KL divergence with shared memory (CUDA)");
}
