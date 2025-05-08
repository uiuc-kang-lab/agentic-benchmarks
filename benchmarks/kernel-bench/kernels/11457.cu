#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction using shuffle intrinsic
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized kernel for KL divergence calculation with minimized warp divergence
__global__ void kl_div_kernel_no_divergence(
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n) {

    float local_sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Use grid-stride loop to evenly distribute the workload
    for (int i = idx; i < n; i += stride) {
        float lp = log_predictions[i];
        float t  = targets[i];
        local_sum += expf(lp) - t * lp;
    }

    // Perform warp-level reduction
    local_sum = warpReduceSum(local_sum);

    // Each warp's lane 0 writes its partial sum into shared memory
    __shared__ float shared[32];  // 32 is safe for blocks up to 1024 threads
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        shared[warpId] = local_sum;
    }
    __syncthreads();

    // Final reduction of partial sums from each warp
    float block_sum = 0.0f;
    if (threadIdx.x < warpSize) {
        // First warp handles the reduction of all partial sums
        block_sum = (threadIdx.x < ((blockDim.x + warpSize - 1) / warpSize)) ? shared[threadIdx.x] : 0.0f;
        block_sum = warpReduceSum(block_sum);
        
        // Only thread 0 needs to write the final result
        if (threadIdx.x == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// CUDA forward function
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    kl_div_kernel_no_divergence<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward with minimized warp divergence (CUDA)");
}
