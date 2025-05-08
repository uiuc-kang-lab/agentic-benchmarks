/*
 * This CUDA kernel combines the unrolled strided loop from Kernel 1 and efficient warp-level
 * reduction with shared memory from Kernel 2. It processes multiple elements per thread to
 * enhance memory coalescing, uses __ldg for read-only global memory access, and then
 * performs warp-level reduction via shuffle intrinsics followed by block-level reduction.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Define constants
constexpr int WARP_SIZE = 32;
constexpr int ELEMENTS_PER_THREAD = 8;  // Process multiple elements per thread

__global__ void efficient_strided_warp_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Calculate global thread id and total threads
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    // Compute number of iterations (strides) for each thread
    const int elements_per_stride = total_threads * ELEMENTS_PER_THREAD;
    const int num_strides = (n + elements_per_stride - 1) / elements_per_stride;

    float sum = 0.0f;

    // Process elements in a strided and unrolled manner
    for (int stride = 0; stride < num_strides; stride++) {
        const int base_idx = stride * elements_per_stride;
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            int idx = base_idx + tid + i * total_threads;
            if (idx < n) {
                // Using __ldg for read-only caching
                float log_pred = __ldg(log_predictions + idx);
                float target = __ldg(targets + idx);
                sum += expf(log_pred) - target * log_pred;
            }
        }
    }

    // Warp-level reduction using shuffle down
    unsigned int mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Identify lane and warp within the block
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    // Use shared memory to store partial sums from each warp
    extern __shared__ float warp_sums[];  // Size should be (blockDim.x / WARP_SIZE)
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // First warp performs block-level reduction
    if (warp_id == 0) {
        float block_sum = (lane_id < (blockDim.x / WARP_SIZE)) ? warp_sums[lane_id] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        if (lane_id == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// Host function to launch the kernel

torch::Tensor efficient_strided_warp_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Configure launch parameters
    const int threads = 256;
    const int min_elements_per_block = threads * ELEMENTS_PER_THREAD;
    const int desired_blocks = (n + min_elements_per_block - 1) / min_elements_per_block;
    // Use a higher maximum block count for higher occupancy
    const int blocks = std::min(desired_blocks, 512);

    const int warps_per_block = threads / WARP_SIZE;
    const int shared_mem = warps_per_block * sizeof(float);

    efficient_strided_warp_kl_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &efficient_strided_warp_kl_forward, "Combined efficient strided warp-reduced KL divergence (CUDA)");
}
