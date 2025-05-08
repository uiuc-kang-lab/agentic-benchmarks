#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define UNROLL_FACTOR 4

// Device function for Smooth L1 (Huber) Loss computation
__device__ inline float huber_loss(float diff) {
    float abs_diff = fabsf(diff);
    return (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
}

// Device function to perform warp-level reduction using shuffle-down operations
__device__ inline float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Combined and optimized CUDA kernel leveraging loop unrolling and warp-level reduction
__global__ void smooth_l1_loss_combined_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    float thread_sum = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    // Each thread processes UNROLL_FACTOR elements at a time
    int stride = total_threads * UNROLL_FACTOR;

    // Unrolled loop processing multiple elements per iteration
    int i = tid * UNROLL_FACTOR;
    for (; i <= n_elements - UNROLL_FACTOR; i += stride) {
        float diff0 = predictions[i] - targets[i];
        float diff1 = predictions[i + 1] - targets[i + 1];
        float diff2 = predictions[i + 2] - targets[i + 2];
        float diff3 = __ldg(&predictions[i + 3]) - targets[i + 3];

        thread_sum += huber_loss(diff0) + huber_loss(diff1) + huber_loss(diff2) + huber_loss(diff3);
    }

    // Process any remaining elements individually
    for (; i < n_elements; i++) {
        float diff = predictions[i] - targets[i];
        thread_sum += huber_loss(diff);
    }

    // Perform warp-level reduction
    thread_sum = warpReduceSum(thread_sum);

    // Shared memory for block-level reduction (one value per warp)
    __shared__ float shared_data[32];  // assuming a maximum of 32 warps per block
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        shared_data[warpId] = thread_sum;
    }
    __syncthreads();

    // First warp reduces the per-warp sums
    if (warpId == 0) {
        thread_sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared_data[lane] : 0.0f;
        thread_sum = warpReduceSum(thread_sum);
    }

    // Thread 0 atomically adds the block's contribution to the output (averaged over n_elements)
    if (threadIdx.x == 0) {
        atomicAdd(output, thread_sum / n_elements);
    }
}

// Host function to set up and launch the kernel
torch::Tensor smooth_l1_loss_cuda(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(
        predictions.sizes() == targets.sizes(),
        "Input tensors must have the same shape"
    );
    TORCH_CHECK(
        predictions.is_contiguous() && targets.is_contiguous(),
        "Input tensors must be contiguous"
    );
    TORCH_CHECK(
        predictions.device().is_cuda() && targets.device().is_cuda(),
        "Inputs must be CUDA tensors"
    );

    int n = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    const int grid_size = (n + block_size * UNROLL_FACTOR - 1) / (block_size * UNROLL_FACTOR);

    smooth_l1_loss_combined_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda, "Smooth L1 Loss Combined Kernel (CUDA)");
}
