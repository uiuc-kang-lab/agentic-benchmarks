#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Define block size for 1D kernel
static const int BLOCK_SIZE = 256;

// An optimized kernel that computes the Mean Squared Error using grid-stride loops and unrolled shared memory reduction
// with warp-level primitives

template <typename scalar_t>
__global__ void efficient_mse_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double smem[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    double thread_sum = 0.0;

    // Grid-stride loop to cover all elements
    for (; idx < num_elements; idx += blockDim.x * gridDim.x) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        thread_sum += diff * diff;
    }

    // Store the partial sum in shared memory
    smem[tid] = thread_sum;
    __syncthreads();

    // Perform reduction in shared memory
    // Unroll reduction loop - can reduce synchronization overhead with warp-level primitives
    if (BLOCK_SIZE >= 512 && tid < 256) {
        smem[tid] += smem[tid + 256];
    }
    __syncthreads();
    if (BLOCK_SIZE >= 256 && tid < 128) {
        smem[tid] += smem[tid + 128];
    }
    __syncthreads();
    if (BLOCK_SIZE >= 128 && tid < 64) {
        smem[tid] += smem[tid + 64];
    }
    __syncthreads();

    // Warp-level reduction without needing __syncthreads() in the final warp
    if (tid < 32) {
        volatile double *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // Atomically add the block's result to the global accumulator
    if (tid == 0) {
        atomicAdd(sum_out, smem[0]);
    }
}

// The forward function that performs the MSE reduction using the optimized kernel

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Determine grid size; optionally limit to a maximum value to avoid launching too many blocks
    int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid_size = std::min(grid_size, 1024);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "efficient_mse_cuda", ([&] {
        efficient_mse_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements);
    }));

    // Calculate the mean of the squared errors
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Efficient Mean Squared Error (MSE) forward (CUDA)");
}
