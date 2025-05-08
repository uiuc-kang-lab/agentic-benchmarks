#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Modular device function for warp-level reduction
template <typename scalar_t>
__device__ inline scalar_t warpReduceSum(scalar_t val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Modular device function for block-level reduction using shared memory
template <typename scalar_t>
__device__ inline scalar_t blockReduceSum(scalar_t val) {
    __shared__ scalar_t shared[32]; // one element per warp
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    
    // First, reduce within the warp
    val = warpReduceSum(val);
    
    // Write reduced value of each warp to shared memory
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // Only the first warp loads the partial sums and reduces them
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < num_warps) ? shared[lane] : scalar_t(0);
    if (warp_id == 0) {
        val = warpReduceSum(val);
    }
    return val;
}


// Kernel for L2 norm reduction using modular device functions
template <typename scalar_t>
__global__ void l2_norm_reduce_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ norms,
    const int C,
    const int stride_C,
    const int outer_stride) {

    const int vec_idx = blockIdx.x; // Each vector (row)
    const int base_offset = vec_idx * outer_stride;
    const int grid_y = gridDim.y; // Number of blocks along the C dimension

    scalar_t partial_sum = 0;
    int start = threadIdx.x + blockIdx.y * blockDim.x;
    int stride = blockDim.x * grid_y;

    // Each thread processes a subset of the vector's elements
    for (int i = start; i < C; i += stride) {
        scalar_t val = input[base_offset + i * stride_C];
        partial_sum += val * val;
    }

    // Reduce the partial sums within the block
    scalar_t block_sum = blockReduceSum<scalar_t>(partial_sum);
    if (threadIdx.x == 0) {
        atomicAdd(&norms[vec_idx], block_sum);
    }
}

// Kernel for L2 normalization using modular structure
template <typename scalar_t>
__global__ void l2_norm_normalize_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ norms,
    const int C,
    const int stride_C,
    const int outer_stride) {

    const int vec_idx = blockIdx.x;
    const int base_offset = vec_idx * outer_stride;

    // Compute the inverse norm with stability epsilon
    scalar_t norm_val = sqrt(norms[vec_idx]) + static_cast<scalar_t>(1e-12);
    scalar_t inv_norm = static_cast<scalar_t>(1.0) / norm_val;

    int grid_y = gridDim.y;
    int start = threadIdx.x + blockIdx.y * blockDim.x;
    int stride = blockDim.x * grid_y;

    // Normalize each element of the vector
    for (int i = start; i < C; i += stride) {
        int index = base_offset + i * stride_C;
        output[index] = input[index] * inv_norm;
    }
}

// Forward function launching the two-phase kernel
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0); // Assumes vectors are contiguous along dim 1

    auto output = torch::empty_like(input);
    auto norms = torch::zeros({total_vectors}, input.options());

    // Grid configuration: each vector is processed in parallel with 2D grid
    const int threads = 256;
    int blocksPerVector = (C + threads - 1) / threads;
    dim3 grid(total_vectors, blocksPerVector);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_reduce_optimized", ([&] {
        l2_norm_reduce_kernel_optimized<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            norms.data_ptr<scalar_t>(),
            C,
            stride_C,
            outer_stride
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_normalize_optimized", ([&] {
        l2_norm_normalize_kernel_optimized<scalar_t><<<grid, threads>>>(
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
    m.def("forward", &forward, "Optimized L2 normalization with efficient indexing");
}
