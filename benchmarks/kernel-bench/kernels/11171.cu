#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel for Smooth L1 (Huber) Loss using shared memory tiling to reduce global memory latency
// This kernel loads contiguous tiles of 'predictions' and 'targets' into shared memory and computes the loss
// for each tile. The per-thread partial sums are then reduced using shared memory reduction and finally
// atomically added to the global output.

__global__ void smooth_l1_loss_shared_tile(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    // Dynamically allocated shared memory: first blockDim.x for predictions, next blockDim.x for targets
    extern __shared__ float shared_buf[];  // size = 2 * blockDim.x * sizeof(float)
    float* s_pred = shared_buf;              // shared memory for predictions
    float* s_targ = shared_buf + blockDim.x;   // shared memory for targets

    float thread_sum = 0.0f;

    // Grid-stride loop over tiles of size blockDim.x
    // Each iteration of this loop loads one tile of data into shared memory
    for (int tile_start = blockIdx.x * blockDim.x; tile_start < n_elements; tile_start += gridDim.x * blockDim.x) {
        int idx = tile_start + threadIdx.x;  // global index for this tile
        // Load data into shared memory if within bounds, else pad with zero
        if (idx < n_elements) {
            s_pred[threadIdx.x] = predictions[idx];
            s_targ[threadIdx.x] = targets[idx];
        } else {
            s_pred[threadIdx.x] = 0.0f;
            s_targ[threadIdx.x] = 0.0f;
        }
        __syncthreads();  // Ensure the tile is fully loaded

        // Compute the loss for the element in the current tile
        if (idx < n_elements) {
            float diff = s_pred[threadIdx.x] - s_targ[threadIdx.x];
            float abs_diff = fabsf(diff);
            thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
        }
        __syncthreads();  // Ensure all threads finish processing this tile before loading the next
    }

    // Reduce partial results within the block using shared memory reduction
    __shared__ float s_sum[256];  // Assumes blockDim.x <= 256
    s_sum[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // The first thread in the block adds the block's contribution
    if (threadIdx.x == 0) {
        atomicAdd(output, s_sum[0] / n_elements);
    }
}

// Host function wrapper
torch::Tensor smooth_l1_loss_cuda_shared_tile(
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
    int grid_size = (n + block_size - 1) / block_size;
    if (grid_size == 0) {
        grid_size = 1;
    }

    // Allocate dynamic shared memory: 2 arrays of block_size floats
    size_t shared_mem_size = 2 * block_size * sizeof(float);

    smooth_l1_loss_shared_tile<<<grid_size, block_size, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda_shared_tile, "Smooth L1 Loss (CUDA) with shared memory tiling");
}
