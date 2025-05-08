#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Inline function to compute the KL divergence contribution per element
__device__ inline float compute_kldiv(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

// Warp-level reduction using shuffle instructions
__device__ inline float warp_reduce_sum(float val) {
    // Use full warp mask (0xffffffff) for active lanes
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// CUDA kernel using grid-stride loop to process workloads larger than available threads
__global__ void kl_div_kernel_stride(const float* __restrict__ log_predictions,
                                      const float* __restrict__ targets,
                                      float* __restrict__ output,
                                      const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;

    // Grid-stride loop: each thread processes multiple elements, verifying boundaries
    for (int i = idx; i < n; i += stride) {
        local_sum += compute_kldiv(log_predictions[i], targets[i]);
    }

    // Reduce within the warp using shuffle operations
    local_sum = warp_reduce_sum(local_sum);

    // Use shared memory to accumulate warp-level results per block
    __shared__ float shared_sum[32]; // Maximum 32 warps per block
    int lane = threadIdx.x & 31;       // threadIdx.x % 32
    int warpId = threadIdx.x >> 5;       // threadIdx.x / 32
    if (lane == 0) {
        shared_sum[warpId] = local_sum;
    }
    __syncthreads();

    // First warp final reduction over the block's partial sums
    int numWarps = (blockDim.x + 31) / 32;
    if (threadIdx.x < numWarps) {
        float sum = shared_sum[threadIdx.x];
        sum = warp_reduce_sum(sum);
        if (threadIdx.x == 0) {
            atomicAdd(output, sum);
        }
    }
}

// Host function wrapping the kernel as a PyTorch CUDA extension
torch::Tensor kl_div_cuda_forward_stride(torch::Tensor log_predictions,
                                           torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    // Calculate blocks based on total elements; grid-stride loop handles large n correctly
    const int blocks = (n + threads - 1) / threads;

    kl_div_kernel_stride<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_stride, "KLDiv forward with grid-stride loop (CUDA)");
}
