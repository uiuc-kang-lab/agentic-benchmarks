#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define warp size constant
constexpr int WARP_SIZE = 32;

// This kernel ensures memory coalescing by aligning global memory accesses.
// It loads data in a vectorized manner (using float4) so that threads in a warp access consecutive memory locations.
// The kernel computes the partial KL divergence (exp(log_predictions) - targets*log_predictions) and uses warp-level reductions
// (via __shfl_down_sync) followed by a block-level reduction in shared memory before accumulating the result atomically.

__global__ void aligned_coalesced_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    float sum = 0.0f;

    // Process main part using vectorized loads (float4) for coalesced access
    int n_vec = n / 4;  // Number of complete groups of 4 floats
    for (int i = tid; i < n_vec; i += total_threads) {
        // Each thread reads a float4 from contiguous memory
        float4 lp = __ldg(reinterpret_cast<const float4*>(log_predictions) + i);
        float4 t  = __ldg(reinterpret_cast<const float4*>(targets) + i);
        sum += expf(lp.x) - t.x * lp.x;
        sum += expf(lp.y) - t.y * lp.y;
        sum += expf(lp.z) - t.z * lp.z;
        sum += expf(lp.w) - t.w * lp.w;
    }

    // Handle remaining elements that don't fit into a group of 4
    int tail_start = n_vec * 4;
    for (int i = tail_start + tid; i < n; i += total_threads) {
        float lp = __ldg(log_predictions + i);
        float t  = __ldg(targets + i);
        sum += expf(lp) - t * lp;
    }

    // Warp-level reduction using shuffle intrinsics
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Shared memory for block-level reduction: one entry per warp
    extern __shared__ float sdata[];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction: First warp in block reduces the per-warp sums
    if (threadIdx.x < (blockDim.x / WARP_SIZE)) {
        float warp_sum = sdata[threadIdx.x];
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

// Host function to launch the kernel
torch::Tensor aligned_coalesced_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    // Determine number of groups processed in vectorized mode
    int n_vec = n / 4;
    int blocks = (n_vec + threads - 1) / threads;
    if (blocks > 256) blocks = 256; // Limit block count to ensure sufficient work per block
    
    // Shared memory: one float per warp
    int shared_mem = (threads / WARP_SIZE) * sizeof(float);

    aligned_coalesced_kl_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &aligned_coalesced_kl_forward, "Aligned & Coalesced KL divergence (CUDA)");
}
