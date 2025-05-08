#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block size to ensure enough threads per block
static const int BLOCK_SIZE = 256;

// This kernel minimizes warp divergence by using a uniform loop iteration count for all threads
// and by computing a validity mask to avoid conditional branches within the inner loop.

template <typename scalar_t>
__global__ void warp_uniform_mse_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    // Compute a unique global thread index
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Compute the number of iterations needed, uniformly for all threads
    int num_iterations = (num_elements + stride - 1) / stride;
    double thread_sum = 0.0;

    // Loop a fixed number of iterations to ensure uniform control flow within warps
    for (int i = 0; i < num_iterations; i++) {
        int idx = global_id + i * stride;
        // Compute a validity mask (1 if idx is in-bound, 0 otherwise)
        int valid = (idx < num_elements);
        // Use a safe index for out-of-bound threads; the result will be multiplied by 0
        int safe_idx = valid ? idx : 0;
        
        // Load values using __ldg for coalesced memory access
        double pred_val = static_cast<double>(__ldg(&preds[safe_idx]));
        double tgt_val  = static_cast<double>(__ldg(&tgts[safe_idx]));
        double diff = pred_val - tgt_val;
        // Multiply by validity mask to avoid conditional branch
        thread_sum += valid * (diff * diff);
    }

    // Warp-level reduction using shuffle instructions; this code executes uniformly across all threads
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Write each warp's partial sum into shared memory
    __shared__ double warp_sums[BLOCK_SIZE / 32];
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // First warp reduces the partial sums from all warps in the block
    if (threadIdx.x < BLOCK_SIZE / 32) {
        double sum = warp_sums[threadIdx.x];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(sum_out, sum);
        }
    }
}

// Host function invoked from Python via Pybind11

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    // Create an accumulator tensor using double precision for accuracy
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Determine grid size; limit maximum blocks to avoid oversubscription
    int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid_size = grid_size > 1024 ? 1024 : grid_size;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "warp_uniform_mse_cuda", ([&] {
        warp_uniform_mse_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    // Compute the final mean squared error
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA) with minimized warp divergence");
}
