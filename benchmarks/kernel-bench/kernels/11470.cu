#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Improved warp-level reduction using shuffle intrinsic
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel for KL divergence optimized with thread and block indexing
__global__ void kl_div_kernel_thread_block_opt(
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n) {

    // Utilize correct and efficient indexing
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    // Use grid-stride loop for workload division
    for (int i = tid; i < n; i += total_threads) {
        float lp = log_predictions[i];
        float t  = targets[i];
        local_sum += expf(lp) - t * lp;
    }

    // Use warp-level reduction
    local_sum = warpReduceSum(local_sum);

    // Shared memory reduction
    __shared__ float shared[32];  // 32 is safe for up to 1024 threads per block
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        shared[warpId] = local_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (threadIdx.x < warpSize) {
        if (threadIdx.x < (blockDim.x / warpSize)) {
            block_sum = shared[threadIdx.x];
        }
        block_sum = warpReduceSum(block_sum);
    }

    // Atomic add to output
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

// CUDA forward function
torch::Tensor kl_div_cuda_forward_thread_block_opt(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 512;  // Increased thread count for better occupancy
    const int blocks = min((n + threads - 1) / threads, 65535); // Ensuring block limit

    kl_div_kernel_thread_block_opt<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_thread_block_opt, "KL divergence forward optimized with thread and block indexing (CUDA)");
}
