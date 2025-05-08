#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Kernel using shared memory tiling for predictions and targets
// This kernel loads contiguous tiles of data into shared memory, reducing global memory latency
// and then computes the squared differences. Reduction is performed in shared memory before
// atomic accumulation into the global sum.

template <typename scalar_t>
__global__ void mse_shared_forward_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    // Shared memory tiles for predictions and targets
    __shared__ scalar_t pred_tile[BLOCK_SIZE];
    __shared__ scalar_t tgt_tile[BLOCK_SIZE];

    double local_sum = 0.0;
    int tid = threadIdx.x;
    
    // Each block processes multiple contiguous tiles
    // Compute starting offset for this block's tiles
    for (int base = blockIdx.x * blockDim.x; base < num_elements; base += blockDim.x * gridDim.x) {
        int index = base + tid;
        // Load data into shared memory if within bounds
        if (index < num_elements) {
            pred_tile[tid] = preds[index];
            tgt_tile[tid] = tgts[index];
        }
        __syncthreads();  // ensure data is loaded before usage
        
        // Each thread computes its contribution using the tile data
        if (index < num_elements) {
            double diff = static_cast<double>(pred_tile[tid]) - static_cast<double>(tgt_tile[tid]);
            local_sum += diff * diff;
        }
        __syncthreads();  // ensure all threads have finished using shared memory
    }

    // Reduction within the block using shared memory for accumulation
    __shared__ double sdata[BLOCK_SIZE];
    sdata[tid] = local_sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Atomic add the block's result to the global accumulator
    if (tid == 0) {
        atomicAdd(sum_out, sdata[0]);
    }
}

// Host function that dispatches the CUDA kernel
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    // Use double precision for accumulation to ensure correctness
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    const int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_shared_forward_cuda", [&] {
        mse_shared_forward_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    });

    // Compute final mean by dividing the accumulated squared error by the number of elements
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA) using shared memory tiling");
}
