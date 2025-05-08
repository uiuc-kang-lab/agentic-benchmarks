#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Define the tile size for loading data into shared memory and the block size
#define TILE_SIZE 1024
#define BLOCK_SIZE 256

// CUDA kernel that leverages shared memory to preload tiles of data from global memory
// The kernel processes the input in tiles, computing squared differences from shared memory,
// and then reduces the result within each block before accumulating into a global sum using atomicAdd.

template <typename scalar_t>
__global__ void tiled_shared_mse_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    // Shared memory buffers for a tile of predictions and targets
    __shared__ scalar_t s_preds[TILE_SIZE];
    __shared__ scalar_t s_tgts[TILE_SIZE];
    // Shared memory for block-level reduction
    __shared__ double sdata[BLOCK_SIZE];

    double block_sum = 0.0;

    // Each block processes multiple tiles in a grid-stride loop
    for (int64_t tile_start = blockIdx.x * TILE_SIZE; tile_start < num_elements; tile_start += gridDim.x * TILE_SIZE) {
        // Determine the number of elements in this tile (may be less than TILE_SIZE at the end)
        int tile_len = (num_elements - tile_start < TILE_SIZE) ? (num_elements - tile_start) : TILE_SIZE;

        // Load tile data from global memory into shared memory
        for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
            s_preds[i] = preds[tile_start + i];
            s_tgts[i] = tgts[tile_start + i];
        }
        __syncthreads();

        // Each thread computes the squared difference for its assigned indices within the tile
        double local_sum = 0.0;
        for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
            double diff = static_cast<double>(s_preds[i]) - static_cast<double>(s_tgts[i]);
            local_sum += diff * diff;
        }

        // Store the local sum into shared memory for reduction
        sdata[threadIdx.x] = local_sum;
        __syncthreads();

        // Intra-block reduction using shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }
        
        // Thread 0 adds the tile's partial result to the block's running sum
        if (threadIdx.x == 0) {
            block_sum += sdata[0];
        }
        __syncthreads(); // Ensure all threads are done before processing the next tile
    }

    // The first thread of the block atomically adds the block's total sum to the global accumulator
    if (threadIdx.x == 0) {
        atomicAdd(sum_out, block_sum);
    }
}

// Host function called from Python to launch the CUDA kernel

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Calculate grid size based on the number of tiles, capped to avoid launching too many blocks
    int grid_size = (num_elements + TILE_SIZE - 1) / TILE_SIZE;
    grid_size = std::min(grid_size, 1024);

    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(grid_size);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "tiled_shared_mse_cuda", ([&] {
        tiled_shared_mse_kernel<scalar_t><<<grid_dim, block_dim>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    // Compute the final MSE by dividing the accumulated sum by the number of elements
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tiled Shared Memory MSE forward (CUDA)");
}
