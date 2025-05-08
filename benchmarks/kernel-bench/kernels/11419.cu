#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute the KL divergence loss for a single element
__device__ inline float compute_loss(const float log_pred, const float target) {
    return expf(log_pred) - target * log_pred;
}

// Device function to perform warp-level reduction using shuffle operations
__device__ inline float warp_reduce_sum(float val) {
    // Unroll reduction within a warp (assuming warp size 32)
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// CUDA kernel for KL divergence computation using modular device functions
__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int grid_stride = blockDim.x * gridDim.x;

    float sum = 0.0f;

    // Grid-stride loop for processing multiple elements per thread
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += grid_stride) {
        sum += compute_loss(log_predictions[idx], targets[idx]);
    }

    // Perform warp-level reduction
    sum = warp_reduce_sum(sum);

    // Allocate shared memory to store partial sums from each warp
    extern __shared__ float shared_warp[];
    if (lane_id == 0) {
        shared_warp[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction: first warp reduces the partial sums
    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x / warp_size)) ? shared_warp[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
}

// Host function that launches the CUDA kernel
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Configure launch parameters
    const int threads = 256;
    const int blocks = min(256, (n + threads - 1) / threads);
    const int warps_per_block = threads / 32;
    const int shared_mem = warps_per_block * sizeof(float);

    kl_div_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}
