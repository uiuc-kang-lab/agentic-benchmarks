#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block size
static const int BLOCK_SIZE = 256;

// CUDA kernel that minimizes warp divergence by precomputing iteration counts
// and using a uniform loop without per-iteration conditionals

template <typename scalar_t>
__global__ void mse_forward_kernel_no_divergence(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    // Global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Total number of threads in the grid
    int stride = blockDim.x * gridDim.x;

    // Each thread computes its iteration count in a uniform way
    int iterations = 0;
    if (tid < num_elements) {
        int full_iters = num_elements / stride;
        int remainder   = num_elements % stride;
        // Use a branchless style: (tid < remainder) evaluates to 1 when true, 0 when false
        int extra = (tid < remainder);
        iterations = full_iters + extra;
    }

    double thread_sum = 0.0;
    int base = tid;
    // Loop the precomputed number of times without additional conditionals
    for (int i = 0; i < iterations; i++) {
        int index = base + i * stride;  // Guaranteed to be valid by construction
        double diff = static_cast<double>(__ldg(&preds[index])) - static_cast<double>(__ldg(&tgts[index]));
        thread_sum += diff * diff;
    }

    // Warp-level reduction using shuffle intrinsics to avoid shared memory if possible
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Use shared memory for block-level reduction
    __shared__ double warp_sums[BLOCK_SIZE / 32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;  // threadIdx.x / 32
    if (lane == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // First warp performs final reduction
    if (threadIdx.x < (BLOCK_SIZE / 32)) {
        double block_sum = warp_sums[threadIdx.x];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(sum_out, block_sum);
        }
    }
}

// Host function to launch the CUDA kernel

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    // Create an accumulator tensor in double precision
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_no_divergence", ([&] {
        mse_forward_kernel_no_divergence<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    // Final mean: accumulated sum divided by number of elements
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MSE forward (CUDA) minimizing warp divergence via uniform iteration counts");
}
