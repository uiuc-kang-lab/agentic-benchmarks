#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;

template <typename scalar_t>
__global__ void mse_forward_kernel_uniform(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    // Each thread accumulates its own sum in double precision
    double thread_sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Compute uniform number of iterations for all threads
    int n_iter = (num_elements + stride - 1) / stride;

    // Loop uniformly for all threads
    for (int iter = 0; iter < n_iter; iter++) {
        int i = idx + iter * stride;
        // For iterations guaranteed in-bound, avoid a conditional
        if (iter < n_iter - 1) {
            // i is guaranteed to be valid
            double diff = static_cast<double>(preds[i]) - static_cast<double>(tgts[i]);
            thread_sum += diff * diff;
        } else {
            // Last iteration: check boundary to avoid out-of-bound access
            if (i < num_elements) {
                double diff = static_cast<double>(preds[i]) - static_cast<double>(tgts[i]);
                thread_sum += diff * diff;
            }
        }
    }

    // Perform warp-level reduction using shuffle instructions (uniform control flow within a warp)
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Each warp's leader writes its reduced sum to shared memory
    int lane = threadIdx.x & 31;
    __shared__ double warp_sums[BLOCK_SIZE / 32];
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // First warp performs block-level reduction on warp sums
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

// Host function launching the kernel

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    
    // Use double precision accumulator to maintain numerical accuracy
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda_uniform", ([&] {
        mse_forward_kernel_uniform<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    // Final mean squared error: divide accumulated sum by number of elements
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MSE forward (CUDA) with uniform iteration to minimize warp divergence");
}
