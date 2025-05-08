#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction using __shfl_down_sync
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void smooth_l1_loss_kernel_warp(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    // Each thread processes a strided range of elements
    for (int i = idx; i < n_elements; i += stride) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        if (abs_diff < 1.0f) {
            thread_sum += 0.5f * diff * diff;
        } else {
            thread_sum += abs_diff - 0.5f;
        }
    }

    // Intra-warp reduction using warp-level primitives
    thread_sum = warp_reduce_sum(thread_sum);

    // Each warp's first lane writes its reduced sum into a small shared memory array
    __shared__ float warpSums[32]; // Maximum 32 warps per block (for blockDim.x <= 1024)
    int lane = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x >> 5;  // Equivalent to threadIdx.x / warpSize
    if (lane == 0) {
        warpSums[warpId] = thread_sum;
    }
    __syncthreads();

    // Final reduction performed by the first warp
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < numWarps) {
        float sum = warpSums[threadIdx.x];
        sum = warp_reduce_sum(sum);
        if (threadIdx.x == 0) {
            // Divide by total number of elements to compute the average
            atomicAdd(output, sum / n_elements);
        }
    }
}

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
    const int grid_size = (n + block_size - 1) / block_size;

    smooth_l1_loss_kernel_warp<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda, "Smooth L1 Loss (CUDA) with warp-level primitives");
}
