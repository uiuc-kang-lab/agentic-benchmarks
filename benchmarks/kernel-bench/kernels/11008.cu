#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 512;  // Increased block size for better occupancy

template <typename scalar_t>
__global__ void mse_forward_kernel_optimized(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements,
    const int num_blocks
) {
    __shared__ double shm[BLOCK_SIZE];
    
    // Calculate global thread index
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * num_blocks;
    
    // Initialize shared memory
    shm[threadIdx.x] = 0.0;
    
    // Process elements with coalesced memory access
    double local_sum = 0.0;
    for (int idx = gid; idx < num_elements; idx += stride) {
        const double pred = static_cast<double>(preds[idx]);
        const double tgt = static_cast<double>(tgts[idx]);
        const double diff = pred - tgt;
        local_sum += diff * diff;
    }
    
    // Store in shared memory
    shm[threadIdx.x] = local_sum;
    __syncthreads();
    
    // Reduction within block using sequential addressing
    #pragma unroll
    for (int offset = BLOCK_SIZE/2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shm[threadIdx.x] += shm[threadIdx.x + offset];
        }
        __syncthreads();
    }
    
    // Single atomic add per block
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
    
    // Calculate optimal grid size based on SM count
    int device_id;
    cudaGetDevice(&device_id);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
    const int num_blocks = sm_count * 4; // 4 blocks per SM for good occupancy
    
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", [&] {
        mse_forward_kernel_optimized<scalar_t><<<num_blocks, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements,
            num_blocks
        );
    });

    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA)");
}