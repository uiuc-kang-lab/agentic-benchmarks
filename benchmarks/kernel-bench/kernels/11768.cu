#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define constants
constexpr int WARP_SIZE = 32;
constexpr int ELEMENTS_PER_THREAD = 8;

// Combined efficient kernel: grid-stride loop with unrolling and warp-level reduction
__global__ void efficient_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Global thread index and total threads
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    float sum = 0.0f;

    // Process multiple elements per thread using a grid-stride loop with unrolling
    // Each iteration covers ELEMENTS_PER_THREAD elements
    for (int base_idx = tid; base_idx < n; base_idx += total_threads * ELEMENTS_PER_THREAD) {
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            int idx = base_idx + i * total_threads;
            if (idx < n) {
                // Use __ldg for read-only cache load
                float log_pred = __ldg(log_predictions + idx);
                float target = __ldg(targets + idx);
                sum += expf(log_pred) - target * log_pred;
            }
        }
    }

    // Perform warp-level reduction using shuffle
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Allocate shared memory for warp-level partial sums
    extern __shared__ float shared_warp[]; // size = (blockDim.x / WARP_SIZE)
    if (lane_id == 0) {
        shared_warp[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction by the first warp
    if (warp_id == 0) {
        // Each thread in the first warp loads one warp's partial sum
        sum = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? shared_warp[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // Only one thread adds the block's sum into the global output
        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
}

// Forward function that sets up kernel launch parameters and returns averaged KL divergence

torch::Tensor efficient_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    // Initialize output tensor
    auto output = torch::zeros({1}, log_predictions.options());

    // Set launch configuration
    const int threads = 256;
    const int min_elements_per_block = threads * ELEMENTS_PER_THREAD;
    const int desired_blocks = (n + min_elements_per_block - 1) / min_elements_per_block;
    const int max_blocks = 256;  // Limit maximum blocks to ensure enough work per block
    const int blocks = (desired_blocks < max_blocks) ? desired_blocks : max_blocks;

    // Shared memory size: one float per warp
    const int shared_mem = (threads / WARP_SIZE) * sizeof(float);

    efficient_kl_div_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    // Return average KL divergence (sum divided by number of elements)
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &efficient_kl_div_forward, "Efficient KL divergence forward (CUDA)");
}
