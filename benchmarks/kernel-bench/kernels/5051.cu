#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction using shuffle instructions
template <typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction with minimal synchronizations
template <typename scalar_t>
__device__ __forceinline__ scalar_t blockReduceSum(scalar_t val) {
    // If the block fits within one warp, no need for shared memory or sync
    if (blockDim.x <= warpSize) {
        return warpReduceSum(val);
    }
    
    // Otherwise, use shared memory
    __shared__ scalar_t shared[32]; // one element per warp
    int lane = threadIdx.x & (warpSize - 1);
    int wid = threadIdx.x / warpSize;
    
    // Each warp performs partial reduction
    val = warpReduceSum(val);

    // Write reduced value of each warp to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    // Synchronize only if there are multiple warps
    __syncthreads();

    // Let the first warp load the partial sums
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < numWarps) {
        val = shared[lane];
    } else {
        val = 0;
    }
    
    // Only threads in the first warp participate in final reduction
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// Kernel for L2 norm reduction with synchronization optimized
template <typename scalar_t>
__global__ void l2_norm_reduce_kernel_sync_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ norms,
    const int C,
    const int stride_C,
    const int outer_stride) {

    const int vec_idx = blockIdx.x; // Each blockIdx.x corresponds to one vector
    const int base_offset = vec_idx * outer_stride;
    const int grid_y = gridDim.y; // Number of blocks along the C dimension

    scalar_t partial_sum = 0;
    int start = threadIdx.x + blockIdx.y * blockDim.x;
    int stride = blockDim.x * grid_y;
    
    // Compute partial sum of squares for this segment
    for (int i = start; i < C; i += stride) {
        scalar_t val = input[base_offset + i * stride_C];
        partial_sum += val * val;
    }

    // Perform block-wide reduction
    scalar_t block_sum = blockReduceSum<scalar_t>(partial_sum);
    if (threadIdx.x == 0) {
        atomicAdd(&norms[vec_idx], block_sum);
    }
}

// Kernel for L2 normalization phase with synchronization optimized
template <typename scalar_t>
__global__ void l2_norm_normalize_kernel_sync_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ norms,
    const int C,
    const int stride_C,
    const int outer_stride) {
    
    const int vec_idx = blockIdx.x;
    const int base_offset = vec_idx * outer_stride;

    // Compute inverse L2 norm with epsilon for stability
    scalar_t norm_val = sqrt(norms[vec_idx]) + static_cast<scalar_t>(1e-12);
    scalar_t inv_norm = static_cast<scalar_t>(1.0) / norm_val;

    int grid_y = gridDim.y;
    int start = threadIdx.x + blockIdx.y * blockDim.x;
    int stride = blockDim.x * grid_y;
    
    // Apply normalization
    for (int i = start; i < C; i += stride) {
        int index = base_offset + i * stride_C;
        output[index] = input[index] * inv_norm;
    }
}

// Forward function launching the two-phase L2 normalization kernels
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);
    auto norms = torch::zeros({total_vectors}, input.options());

    // Configure grid: each vector processed as a row using a 2D grid
    const int threads = 256;
    int blocksPerVector = (C + threads - 1) / threads;
    dim3 grid(total_vectors, blocksPerVector);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_reduce_sync_optimized", ([&] {
        l2_norm_reduce_kernel_sync_optimized<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            norms.data_ptr<scalar_t>(),
            C,
            stride_C,
            outer_stride
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_normalize_sync_optimized", ([&] {
        l2_norm_normalize_kernel_sync_optimized<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            norms.data_ptr<scalar_t>(),
            C,
            stride_C,
            outer_stride
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization with synchronization optimized");
}
