#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cmath>

// 2D-indexed kernel for Smooth L1 (Huber) Loss
// Uses vectorized loads via float4 and a 2D grid/block mapping for efficient thread scheduling.

__global__ void smooth_l1_loss_2d_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    const int n_elements
) {
    // Compute linear thread index using 2D block and grid indices
    int local_tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int global_tid = blockId * (blockDim.x * blockDim.y) + local_tid;

    // Total number of threads in the grid
    int total_threads = gridDim.x * gridDim.y * (blockDim.x * blockDim.y);

    float thread_sum = 0.0f;

    // Vectorized processing: work in float4 units
    int vec_count = n_elements / 4;
    const float4* pred4 = reinterpret_cast<const float4*>(predictions);
    const float4* targ4 = reinterpret_cast<const float4*>(targets);

    for (int i = global_tid; i < vec_count; i += total_threads) {
        float4 p = __ldg(pred4 + i);
        float4 t = __ldg(targ4 + i);

        // Process p.x, p.y, p.z, p.w
        float diff = p.x - t.x;
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;

        diff = p.y - t.y;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;

        diff = p.z - t.z;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;

        diff = p.w - t.w;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Process remaining scalar elements
    int remainder_start = vec_count * 4;
    for (int i = remainder_start + global_tid; i < n_elements; i += total_threads) {
        float diff = __ldg(predictions + i) - __ldg(targets + i);
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Perform block-level reduction in shared memory
    // Assuming blockDim.x * blockDim.y is fixed (e.g., 16x16 = 256 threads per block)
    __shared__ float shared_mem[256];
    int tId = local_tid;  // local thread index within block
    shared_mem[tId] = thread_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tId < s) {
            shared_mem[tId] += shared_mem[tId + s];
        }
        __syncthreads();
    }

    // The first thread in the block updates the global output
    if (tId == 0) {
        // Each block contributes its partial sum normalized by n_elements
        atomicAdd(output, shared_mem[0] / n_elements);
    }
}

// Host function
// This function sets up a 2D grid mapping to cover the data domain
torch::Tensor smooth_l1_loss_2d(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "Inputs must be CUDA tensors");

    int n = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    // Using a block of 16x16 threads (total 256 threads per block)
    dim3 block(16, 16);
    int total_threads_per_block = block.x * block.y;  // 256

    // Determine grid size in vectorized domain (processing elements as float4)
    int vec_count = n / 4;
    int total_blocks = (vec_count + total_threads_per_block - 1) / total_threads_per_block;

    // Map total_blocks to a 2D grid
    int gridX = (int)ceil(sqrt((float)total_blocks));
    int gridY = (total_blocks + gridX - 1) / gridX;
    dim3 grid(gridX, gridY);

    smooth_l1_loss_2d_kernel<<<grid, block>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_2d, "2D-indexed Smooth L1 Loss (CUDA)");
}
