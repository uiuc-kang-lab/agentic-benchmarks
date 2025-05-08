#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Using a fixed tile size equal to block size for shared memory caching
static const int TILE_SIZE = 256;

template <typename scalar_t>
__global__ void mse_forward_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    // Allocate shared memory for a tile of predictions and targets
    __shared__ scalar_t s_preds[TILE_SIZE];
    __shared__ scalar_t s_tgts[TILE_SIZE];

    double local_sum = 0.0;

    // Grid-stride loop over tiles of input data
    // Each tile covers TILE_SIZE consecutive elements
    for (int tile_start = blockIdx.x * TILE_SIZE;
         tile_start < num_elements;
         tile_start += gridDim.x * TILE_SIZE) {

        int idx = tile_start + threadIdx.x;

        // Load a tile of data from global memory into shared memory
        if (idx < num_elements) {
            s_preds[threadIdx.x] = preds[idx];
            s_tgts[threadIdx.x] = tgts[idx];
        } else {
            // For threads out-of-bound, set to 0 to avoid garbage values
            s_preds[threadIdx.x] = 0;
            s_tgts[threadIdx.x] = 0;
        }
        __syncthreads();

        // Compute squared difference for the element in shared memory
        if (idx < num_elements) {
            double diff = static_cast<double>(s_preds[threadIdx.x]) - static_cast<double>(s_tgts[threadIdx.x]);
            local_sum += diff * diff;
        }
        __syncthreads();
    }

    // Reduction within the block using shared memory
    __shared__ double block_sum[TILE_SIZE];
    block_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = TILE_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Atomic add of the block's contribution to the global accumulator
    if (threadIdx.x == 0) {
        atomicAdd(sum_out, block_sum[0]);
    }
}


torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    // Use double for accumulation to preserve precision
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Launch parameters: one thread per element in a tile
    const int block_size = TILE_SIZE; // 256 threads per block
    const int grid_size = (num_elements + TILE_SIZE - 1) / TILE_SIZE;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", ([&] {
        mse_forward_kernel<scalar_t><<<grid_size, block_size>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    // Compute final mean by dividing the accumulated squared error by num_elements
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA)");
}
