#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Anonymous namespace for device helper functions
namespace {

// Inline function to load a float4 vector using __ldg for efficient read-only caching
__device__ inline float4 load_vector4(const float* __restrict__ ptr, int idx) {
    return __ldg(reinterpret_cast<const float4*>(ptr) + idx);
}

// Inline function to process an element: compute exp(lp) - tt * lp
__device__ inline float process_element(float lp, float tt) {
    return expf(lp) - tt * lp;
}

// Warp-level reduction using shfl_down_sync
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

} // end anonymous namespace

// Optimized KL divergence kernel using vectorized loads,
// shared memory for warp reduction, and grid-stride loops
__global__ void kl_div_kernel_optimized(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Each thread's global id
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int global_idx = blockIdx.x * blockDim.x + tid;

    // Shared memory for storing warp sums
    extern __shared__ float warp_sums[];

    float sum = 0.0f;

    // Process vectorized elements (float4)
    const int n4 = n / 4;  // Number of float4 elements
    int vec_idx = global_idx;
    while (vec_idx < n4) {
        // Load vectorized data
        float4 logp = load_vector4(log_predictions, vec_idx);
        float4 targ = load_vector4(targets, vec_idx);

        // Unroll the 4 components
        sum += process_element(logp.x, targ.x);
        sum += process_element(logp.y, targ.y);
        sum += process_element(logp.z, targ.z);
        sum += process_element(logp.w, targ.w);

        vec_idx += gridDim.x * blockDim.x;
    }

    // Process remaining scalar elements if n is not multiple of 4
    int scalar_start = n4 * 4;
    int scalar_idx = scalar_start + global_idx;
    while (scalar_idx < n) {
        float lp = __ldg(log_predictions + scalar_idx);
        float tt = __ldg(targets + scalar_idx);
        sum += process_element(lp, tt);
        scalar_idx += gridDim.x * blockDim.x;
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    // Write each warp's partial sum into shared memory
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // First warp performs block-level reduction of the warp sums
    if (warp_id == 0) {
        float block_sum = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);

        // Atomic add to global output
        if (lane == 0)
            atomicAdd(output, block_sum);
    }
}

// CUDA forward function exposed to PyTorch
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks = min((n + threads * 4 - 1) / (threads * 4), 1024);
    const int shared_mem = warps_per_block * sizeof(float);

    kl_div_kernel_optimized<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Optimized KL divergence forward (CUDA)");
}
