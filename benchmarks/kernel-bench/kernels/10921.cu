#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define constants
static const int BLOCK_SIZE = 256;
static const int WARP_SIZE = 32;

// CUDA kernel that computes MSE with memory coalescing and loop unrolling.
// Each thread uses __ldg to load from read‐only global memory, ensuring that
// threads in a warp access consecutive memory locations. The loop is unrolled
// by a factor of 4 to reduce loop overhead and improve instruction throughput.

template <typename scalar_t>
__global__ void mse_forward_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    double sum = 0.0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Loop unrolling factor 4
    // Each thread processes 4 elements per iteration if available.
    for (int i = tid; i + 3 * stride < num_elements; i += stride * 4) {
        double diff0 = static_cast<double>(__ldg(&preds[i])) - static_cast<double>(__ldg(&tgts[i]));
        double diff1 = static_cast<double>(__ldg(&preds[i + stride])) - static_cast<double>(__ldg(&tgts[i + stride]));
        double diff2 = static_cast<double>(__ldg(&preds[i + 2 * stride])) - static_cast<double>(__ldg(&tgts[i + 2 * stride]));
        double diff3 = static_cast<double>(__ldg(&preds[i + 3 * stride])) - static_cast<double>(__ldg(&tgts[i + 3 * stride]));
        sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
    }

    // Process remaining elements that don't fit the unroll factor
    for (int i = tid + ((num_elements - tid) / stride) * stride; i < num_elements; i += stride) {
        double diff = static_cast<double>(__ldg(&preds[i])) - static_cast<double>(__ldg(&tgts[i]));
        sum += diff * diff;
    }

    // Warp-level reduction using shuffle
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Each warp's first thread stores its reduced sum in shared memory
    __shared__ double shared[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        shared[warpId] = sum;
    }
    __syncthreads();

    // First warp reduces the per-warp partial sums
    if (threadIdx.x < BLOCK_SIZE / WARP_SIZE) {
        double warpSum = shared[threadIdx.x];
        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            warpSum += __shfl_down_sync(0xffffffff, warpSum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(sum_out, warpSum);
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
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Determine grid size based on the number of elements
    int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", ([&] {
        mse_forward_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    // Compute the mean by dividing the accumulated sum by the number of elements
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA)");
}
