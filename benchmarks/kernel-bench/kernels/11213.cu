#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// First kernel: Each block computes a partial sum of the Smooth L1 (Huber) Loss
// and writes its result to a temporary global array (block_results).
__global__ void partial_smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* block_results,
    int n_elements
) {
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float thread_sum = 0.0f;

    // Process groups of 4 elements at a time
    for (int i = global_thread_id * 4; i <= n_elements - 4; i += stride * 4) {
        float diff0 = predictions[i] - targets[i];
        float diff1 = predictions[i + 1] - targets[i + 1];
        float diff2 = predictions[i + 2] - targets[i + 2];
        float diff3 = predictions[i + 3] - targets[i + 3];

        float abs_diff0 = fabsf(diff0);
        float abs_diff1 = fabsf(diff1);
        float abs_diff2 = fabsf(diff2);
        float abs_diff3 = fabsf(diff3);

        thread_sum += (abs_diff0 < 1.0f ? 0.5f * diff0 * diff0 : abs_diff0 - 0.5f);
        thread_sum += (abs_diff1 < 1.0f ? 0.5f * diff1 * diff1 : abs_diff1 - 0.5f);
        thread_sum += (abs_diff2 < 1.0f ? 0.5f * diff2 * diff2 : abs_diff2 - 0.5f);
        thread_sum += (abs_diff3 < 1.0f ? 0.5f * diff3 * diff3 : abs_diff3 - 0.5f);
    }

    // Process any remaining elements
    int rem_start = (n_elements / 4) * 4;
    for (int i = rem_start + global_thread_id; i < n_elements; i += stride) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f ? 0.5f * diff * diff : abs_diff - 0.5f);
    }

    // Optimized block-level reduction using warp shuffles
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float sum = thread_sum;
    
    // In-warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Write reduced value of each warp to shared memory
    if ((tid & (warpSize - 1)) == 0) {
        sdata[tid / warpSize] = sum;
    }
    __syncthreads();
    
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (tid < num_warps) {
        sum = sdata[tid];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }
    if (tid == 0) {
        block_results[blockIdx.x] = sum;
    }
}

// Second kernel: Final reduction of block partial sums.
// Aggregates the block_results into a single loss value.
__global__ void final_reduce_kernel(
    float* block_results,
    float* output,
    int num_blocks,
    int n_elements
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float sum = 0.0f;

    // Each thread accumulates multiple block results if necessary
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += block_results[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = sdata[0] / n_elements;
    }
}


// Host function that sets up and launches the two-stage reduction kernels
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
    // Each thread processes 4 elements, so calculate the grid size accordingly
    int grid_size = (n + block_size * 4 - 1) / (block_size * 4);

    auto block_results = torch::zeros({grid_size}, predictions.options());

    // Launch the first kernel to compute partial sums per block
    partial_smooth_l1_loss_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_results.data_ptr<float>(),
        n
    );

    // Launch the second kernel to reduce the block results into the final loss value
    final_reduce_kernel<<<1, block_size, block_size * sizeof(float)>>>(
        block_results.data_ptr<float>(),
        output.data_ptr<float>(),
        grid_size,
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda, "Smooth L1 Loss (CUDA) with Two-Stage Reduction");
}
