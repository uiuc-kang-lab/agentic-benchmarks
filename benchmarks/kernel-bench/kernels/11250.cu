#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel using shared memory and warp-level primitives for reduction
__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    // Each thread accumulates a partial sum over a grid-stride loop
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float sum_val = 0.0f;

    for (int i = idx; i < n_elements; i += stride) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        // Compute Huber (Smooth L1) loss
        float loss = (abs_diff < 1.0f) ? (0.5f * diff * diff) : (abs_diff - 0.5f);
        sum_val += loss;
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xFFFFFFFF;  // full warp mask
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_val += __shfl_down_sync(mask, sum_val, offset);
    }

    // Each warp's lane 0 writes its result into shared memory
    __shared__ float warp_sums[256 / 32];  // Assuming blockDim.x is 256
    int lane = threadIdx.x & 31;         // threadIdx.x % 32
    int warp_id = threadIdx.x >> 5;        // threadIdx.x / 32
    if (lane == 0) {
        warp_sums[warp_id] = sum_val;
    }
    __syncthreads();

    // Final reduction: only the first warp handles the reduction of warp sums
    float block_sum = 0.0f;
    if (threadIdx.x < (blockDim.x / 32)) {
        block_sum = warp_sums[threadIdx.x];
        // Reduce values within the warp
        for (int offset = (blockDim.x / 32) / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
    }

    // Thread 0 adds the block's contribution to the global output
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum / n_elements);
    }
}

// Host function interfacing with PyTorch
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

    int n_elements = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    const int grid_size = (n_elements + block_size - 1) / block_size;

    smooth_l1_loss_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n_elements
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda, "Smooth L1 Loss (CUDA)");
}
