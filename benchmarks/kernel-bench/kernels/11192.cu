#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 32
#define BLOCK_SIZE 256

__global__ void smooth_l1_loss_shared_tiled_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    const int n_elements
) {
    __shared__ float pred_tile[TILE_SIZE * 4];
    __shared__ float targ_tile[TILE_SIZE * 4];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = bid * blockDim.x + tid;
    float thread_sum = 0.0f;

    // Process data in tiles using vectorized loads
    const int n_tiles = (n_elements / 4 + TILE_SIZE - 1) / TILE_SIZE;
    const float4* pred4 = reinterpret_cast<const float4*>(predictions);
    const float4* targ4 = reinterpret_cast<const float4*>(targets);

    for (int tile = 0; tile < n_tiles; tile++) {
        const int tile_offset = tile * TILE_SIZE;
        const int global_idx = tile_offset + tid;
        
        // Load tile into shared memory
        if (global_idx < n_elements/4) {
            float4 p = __ldg(pred4 + global_idx);
            float4 t = __ldg(targ4 + global_idx);
            
            // Store to shared memory
            const int shared_idx = tid * 4;
            pred_tile[shared_idx] = p.x;
            pred_tile[shared_idx + 1] = p.y;
            pred_tile[shared_idx + 2] = p.z;
            pred_tile[shared_idx + 3] = p.w;
            
            targ_tile[shared_idx] = t.x;
            targ_tile[shared_idx + 1] = t.y;
            targ_tile[shared_idx + 2] = t.z;
            targ_tile[shared_idx + 3] = t.w;
        }
        __syncthreads();

        // Process tile data
        const int tile_elements = min(TILE_SIZE * 4, n_elements - tile_offset * 4);
        for (int i = tid; i < tile_elements; i += blockDim.x) {
            float diff = pred_tile[i] - targ_tile[i];
            float abs_diff = fabsf(diff);
            thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
        }
        __syncthreads();
    }

    // Handle remaining elements
    const int vec_processed = (n_elements / 4) * 4;
    for (int i = vec_processed + gid; i < n_elements; i += gridDim.x * blockDim.x) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Block-level reduction
    __shared__ float warp_sums[BLOCK_SIZE/warpSize];
    const int lane = tid % warpSize;
    const int wid = tid / warpSize;

    if (lane == 0) {
        warp_sums[wid] = thread_sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (wid == 0) {
        thread_sum = (tid < blockDim.x/warpSize) ? warp_sums[lane] : 0.0f;
        
        #pragma unroll
        for (int offset = (blockDim.x/warpSize)/2; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }

        if (lane == 0) {
            atomicAdd(output, thread_sum / n_elements);
        }
    }
}

torch::Tensor smooth_l1_loss_shared_tiled(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "Inputs must be CUDA tensors");

    const int n = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    const int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    smooth_l1_loss_shared_tiled_kernel<<<grid_size, BLOCK_SIZE>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_shared_tiled, "Smooth L1 Loss with shared memory tiling");
}