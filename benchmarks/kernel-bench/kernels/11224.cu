#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cmath>

namespace cg = cooperative_groups;

// Fused kernel that computes Smooth L1 Loss using loop unrolling and two-stage reduction
// with grid-wide synchronization via cooperative groups. This fuses the partial block
// reduction and the final global reduction into a single kernel launch.

__global__ void fused_smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* block_results,  // temporary array to store per-block sums
    float* output,         // final output (single element)
    int n_elements
) {
    // Each thread processes multiple elements in a grid stride loop with 4x unrolling
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;
    int total_threads = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    // Process groups of 4 elements at a time
    // Compute the largest multiple of 4 that is less than n_elements
    int n_4 = (n_elements / 4) * 4;
    for (int i = global_id * 4; i < n_4; i += total_threads * 4) {
        float diff0 = predictions[i]     - targets[i];
        float diff1 = predictions[i + 1] - targets[i + 1];
        float diff2 = predictions[i + 2] - targets[i + 2];
        float diff3 = predictions[i + 3] - targets[i + 3];

        float abs0 = fabsf(diff0);
        float abs1 = fabsf(diff1);
        float abs2 = fabsf(diff2);
        float abs3 = fabsf(diff3);

        thread_sum += (abs0 < 1.f ? 0.5f * diff0 * diff0 : abs0 - 0.5f);
        thread_sum += (abs1 < 1.f ? 0.5f * diff1 * diff1 : abs1 - 0.5f);
        thread_sum += (abs2 < 1.f ? 0.5f * diff2 * diff2 : abs2 - 0.5f);
        thread_sum += (abs3 < 1.f ? 0.5f * diff3 * diff3 : abs3 - 0.5f);
    }

    // Process any remaining elements
    for (int i = global_id + n_4; i < n_elements; i += total_threads) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.f ? 0.5f * diff * diff : abs_diff - 0.5f);
    }

    // Intra-block reduction using shared memory
    extern __shared__ float sdata[];  // Size assumed to be blockDim.x
    sdata[tid] = thread_sum;
    __syncthreads();

    // Reduce shared memory in a tree-based manner
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Use warp-level reduction for the last 32 threads
    if (tid < 32) {
        float val = sdata[tid];
        for (int offset = 32; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        sdata[tid] = val;
    }
    __syncthreads();

    // Write the block's partial sum to global memory
    if (tid == 0) {
        block_results[blockIdx.x] = sdata[0];
    }

    // Grid-level synchronization using cooperative groups
    cg::grid_group grid = cg::this_grid();
    grid.sync();

    // Final reduction: one block (block 0) reduces all block_results
    if (blockIdx.x == 0) {
        float final_sum = 0.f;
        int num_blocks = gridDim.x;
        // Each thread in block 0 accumulates over a strided range of block_results
        for (int i = tid; i < num_blocks; i += blockDim.x) {
            final_sum += block_results[i];
        }
        sdata[tid] = final_sum;
        __syncthreads();

        // Reduce the partial sums within block 0
        for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
            if (tid < s)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }

        if (tid < 32) {
            float val = sdata[tid];
            for (int offset = 32; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            sdata[tid] = val;
        }
        __syncthreads();

        if (tid == 0) {
            // Write the final result normalized by n_elements
            output[0] = sdata[0] / n_elements;
        }
    }
}

// Host function that wraps the fused kernel launch
// Note: This kernel uses cooperative groups grid synchronization so it must be
// launched via cudaLaunchCooperativeKernel and requires GPU support for it.

torch::Tensor smooth_l1_loss_cuda(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "Inputs must be CUDA tensors");

    int n = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    // Each thread processes 4 elements so grid size is computed accordingly
    int grid_size = (n + block_size * 4 - 1) / (block_size * 4);

    // Allocate temporary workspace for block-level results
    auto block_results = torch::empty({grid_size}, predictions.options());

    // Prepare kernel arguments
    void* args[] = {
        (void*)&(predictions.data_ptr<float>()[0]),
        (void*)&(targets.data_ptr<float>()[0]),
        (void*)&(block_results.data_ptr<float>()[0]),
        (void*)&(output.data_ptr<float>()[0]),
        (void*)&n
    };

    // Launch the cooperative kernel
    // Shared memory size is block_size * sizeof(float) for intra-block reduction
    cudaError_t launch_status = cudaLaunchCooperativeKernel(
        (void*)fused_smooth_l1_loss_kernel,
        grid_size,
        block_size,
        args,
        block_size * sizeof(float),
        0
    );
    TORCH_CHECK(launch_status == cudaSuccess, "Kernel launch failed");

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda, "Fused Smooth L1 Loss with Cooperative Groups Reduction");
}
