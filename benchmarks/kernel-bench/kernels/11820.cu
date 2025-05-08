#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel to compute KL divergence with minimized warp divergence
__global__ void kl_div_kernel_min_divergence(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* output,
    int n) {

    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int global_thread_id = blockIdx.x * blockSize + tid;
    int stride = blockSize * gridDim.x;
    float sum = 0.0f;

    // Process in groups of 4 elements to enable vectorized loads
    // Compute the largest multiple of 4 less than or equal to n
    int n_vec = n & ~3;  // equivalent to (n / 4) * 4

    // Vectorized loop: All threads execute uniform control flow without divergent branches
    for (int idx = global_thread_id * 4; idx < n_vec; idx += stride * 4) {
        // Use float4 to load 4 contiguous floats at a time
        float4 lpred = *(reinterpret_cast<const float4*>(log_predictions + idx));
        float4 targ  = *(reinterpret_cast<const float4*>(targets + idx));

        sum += __expf(lpred.x) - targ.x * lpred.x;
        sum += __expf(lpred.y) - targ.y * lpred.y;
        sum += __expf(lpred.z) - targ.z * lpred.z;
        sum += __expf(lpred.w) - targ.w * lpred.w;
    }

    // Tail loop: Process remaining elements uniformly without introducing divergent branches
    for (int idx = n_vec + global_thread_id; idx < n; idx += stride) {
        float lp = log_predictions[idx];
        float ta = targets[idx];
        sum += __expf(lp) - ta * lp;
    }

    // Warp-level reduction using shuffle instructions to avoid divergence
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Each warp's first lane stores its result into shared memory
    __shared__ float shared[32];
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction of warp sums by the first warp
    float block_sum = (tid < (blockSize + 31) / 32) ? shared[lane] : 0.0f;
    if (tid < 32) {
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
        }
        if (tid == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// Host wrapper function
torch::Tensor kl_div_cuda_forward_minwarpdiv(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    // Calculate blocks based on processing 4 elements per thread
    const int blocks = min((n + threads * 4 - 1) / (threads * 4), 1024);

    kl_div_kernel_min_divergence<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_minwarpdiv, "KL divergence forward with minimal warp divergence (CUDA)");
}
