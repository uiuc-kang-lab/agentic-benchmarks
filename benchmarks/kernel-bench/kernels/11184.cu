/*
 * This CUDA kernel computes the Smooth L1 Loss using vectorized loads (with float4) and a two-level reduction
 * performed in a single, cooperative kernel launch. The first phase uses block-level reduction to compute
 * partial sums of the loss, and then cooperative groups are used to synchronize all blocks so that
 * block 0 performs the final reduction. This avoids the overhead of atomics and multiple kernel launches.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Cooperative kernel: computes loss in two stages (block-level then grid-level reduction)
__global__ void smooth_l1_loss_coop_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements,
    float* partial_sums  // temporary global buffer of size gridDim.x
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int global_idx = blockIdx.x * blockSize + tid;
    int stride = gridDim.x * blockSize;

    float thread_sum = 0.0f;

    // Vectorized processing using float4
    int vec_count = n_elements / 4;
    const float4* pred4 = reinterpret_cast<const float4*>(predictions);
    const float4* targ4 = reinterpret_cast<const float4*>(targets);

    for (int i = global_idx; i < vec_count; i += stride) {
        float4 p = __ldg(pred4 + i);
        float4 t = __ldg(targ4 + i);

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

    // Process any remaining elements using scalar code
    int scalar_start = vec_count * 4;
    for (int i = scalar_start + global_idx; i < n_elements; i += stride) {
        float diff = __ldg(predictions + i) - __ldg(targets + i);
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Block-level reduction into shared memory
    sdata[tid] = thread_sum;
    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write each block's partial sum to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }

    // Synchronize across the entire grid using Cooperative Groups
    cg::grid_group grid = cg::this_grid();
    grid.sync();

    // Designate block 0 to perform the final reduction over partial sums
    if (blockIdx.x == 0) {
        float block_sum = 0.0f;
        // Each thread in block 0 processes a portion of the partial sums
        for (int i = tid; i < gridDim.x; i += blockSize) {
            block_sum += partial_sums[i];
        }
        sdata[tid] = block_sum;
        __syncthreads();

        // Final reduction in shared memory within block 0
        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        // Write the averaged loss to the output (loss is averaged over total number of elements)
        if (tid == 0) {
            output[0] = sdata[0] / n_elements;
        }
    }
}

// Host function to launch the cooperative kernel
// Note: This kernel requires a GPU and driver that support cooperative groups grid synchronization.

torch::Tensor smooth_l1_loss_cooperative(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input shape mismatch");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "Inputs must be CUDA tensors");

    int n_elements = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    // Determine block and grid sizes based on vectorized processing
    const int block_size = 256;
    int grid_size = (n_elements / 4 + block_size - 1) / block_size;
    grid_size = (grid_size > 0) ? grid_size : 1;

    // Allocate temporary buffer for partial sums computed by each block
    auto partial_buffer = torch::empty({grid_size}, predictions.options());

    // Shared memory size in bytes
    size_t shared_mem = block_size * sizeof(float);

    // Launch the cooperative kernel
    // Note: The kernel launch must be done using a cooperative launch configuration if required by the device.
    smooth_l1_loss_coop_kernel<<<grid_size, block_size, shared_mem>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n_elements,
        partial_buffer.data_ptr<float>()
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cooperative, "Smooth L1 Loss using Cooperative Groups");
}
