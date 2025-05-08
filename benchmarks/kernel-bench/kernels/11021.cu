#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 512;
static const int VECTOR_SIZE = 4;  // Process 4 elements per thread

template <typename scalar_t>
__global__ void mse_forward_kernel_unrolled(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double shm[BLOCK_SIZE + 32];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int grid_stride = gridDim.x * blockDim.x;
    
    // Initialize local sum
    double local_sum = 0.0;
    
    // Vector loading - process 4 elements per iteration
    #pragma unroll
    for (int base_idx = gid * VECTOR_SIZE; base_idx < num_elements; base_idx += grid_stride * VECTOR_SIZE) {
        double diff[VECTOR_SIZE];
        
        // Manually unrolled vector load and computation
        #pragma unroll
        for (int i = 0; i < VECTOR_SIZE; i++) {
            int idx = base_idx + i;
            if (idx < num_elements) {
                diff[i] = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
                local_sum += diff[i] * diff[i];
            }
        }
    }
    
    // Store in shared memory
    shm[tid] = local_sum;
    __syncthreads();
    
    // Reduction with manual unrolling for different power-of-2 sizes
    #pragma unroll
    if (tid < 256) { shm[tid] += shm[tid + 256]; } __syncthreads();
    #pragma unroll
    if (tid < 128) { shm[tid] += shm[tid + 128]; } __syncthreads();
    #pragma unroll
    if (tid < 64) { shm[tid] += shm[tid + 64]; } __syncthreads();
    
    // Warp-level reduction (no sync needed)
    if (tid < 32) {
        volatile double* vmem = shm;
        #pragma unroll
        if (tid < 32) vmem[tid] += vmem[tid + 32];
        #pragma unroll
        if (tid < 16) vmem[tid] += vmem[tid + 16];
        #pragma unroll
        if (tid < 8) vmem[tid] += vmem[tid + 8];
        #pragma unroll
        if (tid < 4) vmem[tid] += vmem[tid + 4];
        #pragma unroll
        if (tid < 2) vmem[tid] += vmem[tid + 2];
        #pragma unroll
        if (tid < 1) vmem[tid] += vmem[tid + 1];
    }
    
    // Single atomic add per block
    if (tid == 0) {
        atomicAdd(sum_out, shm[0]);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    
    // Calculate optimal grid size
    const int sm_count = 108; // H100 SM count
    const int blocks_per_sm = 4;
    const int num_blocks = sm_count * blocks_per_sm;
    
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", [&] {
        mse_forward_kernel_unrolled<scalar_t><<<num_blocks, BLOCK_SIZE>>>(
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