#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define constants
constexpr int WARP_SIZE = 32;
constexpr int ELEMENTS_PER_THREAD = 8;

// Optimized CUDA kernel for KL divergence using grid-stride loop, loop unrolling, and warp-level reduction
__global__ void fast_strided_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    const int total_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.f;

    // Each thread processes multiple elements using a grid-stride loop and unrolling
    for (int stride = 0;; stride++) {
        // Compute base index for this iteration
        int base = tid + stride * total_threads * ELEMENTS_PER_THREAD;
        if (base >= n) break;
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            int idx = base + i * total_threads;
            if (idx < n) {
                // Use __ldg for read-only cache load
                float lp = __ldg(log_predictions + idx);
                float t = __ldg(targets + idx);
                sum += expf(lp) - t * lp;
            }
        }
    }

    // Intra-warp reduction using shuffle
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Allocate shared memory for each warp's result
    extern __shared__ float warp_sums[];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // First thread in each warp writes its result
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }

    __syncthreads();

    // Final reduction: let the first warp reduce the per-warp sums
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    sum = (threadIdx.x < num_warps) ? warp_sums[threadIdx.x] : 0.f;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // The first thread of the block atomically adds the block's sum to the global output
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// Host function to launch the optimized kernel
torch::Tensor fast_strided_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Define kernel launch parameters
    const int threads = 256;
    int blocks = (n + threads * ELEMENTS_PER_THREAD - 1) / (threads * ELEMENTS_PER_THREAD);
    // Optionally limit the number of blocks to ensure sufficient work per block
    const int max_blocks = 256;
    blocks = (blocks < max_blocks) ? blocks : max_blocks;

    // Calculate shared memory size: one float per warp
    int shared_mem = ((threads + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);

    fast_strided_kl_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fast_strided_kl_forward, "Optimized KL divergence (CUDA)");
}
