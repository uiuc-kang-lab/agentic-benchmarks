#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;
static const int WARP_SIZE = 32;

template <typename scalar_t>
__global__ void mse_forward_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double shm[BLOCK_SIZE];
    
    // Align block starting index to warp size
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int grid_stride = blockDim.x * gridDim.x;
    
    // Ensure starting index is aligned to warp size
    int base_idx = (blockIdx.x * blockDim.x + warp_id * WARP_SIZE) + lane_id;
    double thread_sum = 0.0;

    // Coalesced memory access within warps
    #pragma unroll 4
    while (base_idx < num_elements) {
        double diff = static_cast<double>(preds[base_idx]) - static_cast<double>(tgts[base_idx]);
        thread_sum += diff * diff;
        base_idx += grid_stride;
    }

    // Store in shared memory
    shm[threadIdx.x] = thread_sum;
    __syncthreads();

    // Warp-aligned reduction
    if (threadIdx.x < 128) shm[threadIdx.x] += shm[threadIdx.x + 128];
    __syncthreads();
    if (threadIdx.x < 64) shm[threadIdx.x] += shm[threadIdx.x + 64];
    __syncthreads();
    
    // Last warp reduces
    if (threadIdx.x < 32) {
        volatile double* smem = shm;
        if (BLOCK_SIZE >= 64) smem[threadIdx.x] += smem[threadIdx.x + 32];
        if (BLOCK_SIZE >= 32) smem[threadIdx.x] += smem[threadIdx.x + 16];
        if (BLOCK_SIZE >= 16) smem[threadIdx.x] += smem[threadIdx.x + 8];
        if (BLOCK_SIZE >= 8) smem[threadIdx.x] += smem[threadIdx.x + 4];
        if (BLOCK_SIZE >= 4) smem[threadIdx.x] += smem[threadIdx.x + 2];
        if (BLOCK_SIZE >= 2) smem[threadIdx.x] += smem[threadIdx.x + 1];
    }

    if (threadIdx.x == 0) {
        atomicAdd(sum_out, shm[0]);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Ensure grid size is multiple of warp size for better memory alignment
    const int grid_size = ((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE + WARP_SIZE - 1) & ~(WARP_SIZE - 1);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", [&] {
        mse_forward_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    });

    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA)");
}