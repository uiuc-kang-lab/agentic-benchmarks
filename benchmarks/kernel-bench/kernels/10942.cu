#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

static const int BLOCK_SIZE = 256;

template <typename scalar_t>
__global__ void optimized_mse_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double smem[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    double thread_sum = 0.0;

    // Optimized grid-stride loop
    for (; idx < num_elements; idx += blockDim.x * gridDim.x) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        thread_sum += diff * diff;
    }

    smem[tid] = thread_sum;
    __syncthreads();

    // Unrolled warp-centric reduction
    for (int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    // Warp-level reduction
    if (tid < 32) {
        volatile double* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) atomicAdd(sum_out, smem[0]);
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(predictions.numel() == targets.numel(), "Input sizes must match");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid_size = std::min(grid_size, 2048);  // Increased max blocks for better utilization

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "optimized_mse_cuda", ([&] {
        optimized_mse_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    return accumulator.div_(num_elements).to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized MSE forward (CUDA)");
}