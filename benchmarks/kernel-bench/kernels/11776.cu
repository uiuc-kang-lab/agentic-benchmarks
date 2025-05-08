#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define warp size constant
constexpr int WARP_SIZE = 32;

// This kernel leverages vectorized (float4) loads with __ldg to ensure that threads in a warp access consecutive memory locations
// thereby improving memory coalescing. It computes the per-thread partial sum of the KL divergence and then reduces it using warp
// shuffle and shared memory before accumulating the result via atomicAdd.
__global__ void vectorized_aligned_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    int total_threads = gridDim.x * blockDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;

    // Process elements in groups of 4 using vectorized loads
    int num_vec = n / 4;  // number of complete groups of 4
    for (int i = tid; i < num_vec; i += total_threads) {
        // Each thread reads a float4 (16 bytes) ensuring coalesced access
        float4 lp = __ldg(reinterpret_cast<const float4*>(log_predictions) + i);
        float4 t  = __ldg(reinterpret_cast<const float4*>(targets) + i);
        sum += expf(lp.x) - t.x * lp.x;
        sum += expf(lp.y) - t.y * lp.y;
        sum += expf(lp.z) - t.z * lp.z;
        sum += expf(lp.w) - t.w * lp.w;
    }

    // Process any remaining elements that don't fit into a group of 4
    int tail_start = num_vec * 4;
    for (int i = tail_start + tid; i < n; i += total_threads) {
        float lp = log_predictions[i];
        float t  = targets[i];
        sum += expf(lp) - t * lp;
    }

    // Warp-level reduction using shuffle instructions
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Use shared memory to reduce the sums from different warps in the block
    extern __shared__ float shared[];
    int lane = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        shared[warpId] = sum;
    }
    __syncthreads();

    // Final reduction performed by the first warp of the block
    if (threadIdx.x < (blockDim.x / WARP_SIZE)) {
        sum = shared[threadIdx.x];
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(output, sum);
        }
    }
}

// Host function that sets up and launches the kernel
torch::Tensor vectorized_aligned_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    // Compute number of groups (of 4 floats) for vectorized processing
    int num_vec = n / 4;
    int blocks = (num_vec + threads - 1) / threads;
    // Limit blocks if necessary to ensure enough work per block
    blocks = min(blocks, 256);

    // Shared memory: one float per warp
    int shared_mem = (threads / WARP_SIZE) * sizeof(float);

    vectorized_aligned_kl_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &vectorized_aligned_kl_forward, "Vectorized & Aligned KL divergence (CUDA)");
}
