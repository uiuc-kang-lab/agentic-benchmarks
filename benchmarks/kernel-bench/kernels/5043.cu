#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function: Reduce a value across a warp
template <typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Device function: Reduce values across a block using shared memory
template <typename scalar_t>
__device__ __forceinline__ scalar_t blockReduceSum(scalar_t val) {
    __shared__ scalar_t shared[32]; // One element per warp (assuming blockDim.x <= warpSize*32)
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    
    // In-warp reduction
    val = warpReduceSum(val);

    // Write reduced value of this warp to shared memory
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // Only the first warp loads all partial sums and reduces them
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < num_warps) ? shared[lane] : scalar_t(0);
    if (warp_id == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// Device function: Compute partial sum of squared values for a given vector segment
template <typename scalar_t>
__device__ __forceinline__ scalar_t computePartialSum(
    const scalar_t* __restrict__ input,
    int base_offset,
    int C,
    int stride_C,
    int block_y,
    int blockDimX,
    int grid_y) {
    scalar_t sum = 0;
    int start = block_y * blockDimX + threadIdx.x;
    int stride = blockDimX * grid_y;
    for (int i = start; i < C; i += stride) {
        scalar_t val = input[base_offset + i * stride_C];
        sum += val * val;
    }
    return sum;
}

// Kernel: Reduction phase for L2 norm computation using modular device functions
template <typename scalar_t>
__global__ void l2_norm_reduce_kernel_refactored(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ norms,
    int C,
    int stride_C,
    int outer_stride) {

    int vec_idx = blockIdx.x; // Each blockIdx.x corresponds to one vector
    int base_offset = vec_idx * outer_stride;

    // Compute partial sum for this block along dimension C
    scalar_t partial = computePartialSum(input, base_offset, C, stride_C, blockIdx.y, blockDim.x, gridDim.y);

    // Reduce the partial sums across the block
    scalar_t block_sum = blockReduceSum<scalar_t>(partial);
    if (threadIdx.x == 0) {
        atomicAdd(&norms[vec_idx], block_sum);
    }
}

// Kernel: Normalization phase for L2 norm using modular device functions
template <typename scalar_t>
__global__ void l2_norm_normalize_kernel_refactored(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ norms,
    int C,
    int stride_C,
    int outer_stride) {

    int vec_idx = blockIdx.x;
    int base_offset = vec_idx * outer_stride;

    // Calculate the L2 norm, adding a small epsilon for numerical stability
    scalar_t norm_val = sqrt(norms[vec_idx]) + static_cast<scalar_t>(1e-12);
    scalar_t inv_norm = static_cast<scalar_t>(1.0) / norm_val;

    int start = blockIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.y;
    
    // Normalize each element in the vector
    for (int i = start; i < C; i += stride) {
        int index = base_offset + i * stride_C;
        output[index] = input[index] * inv_norm;
    }
}

// Forward function: Launches the two-phase L2 normalization kernels
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");

    int C = input.size(1);
    int total_vectors = input.numel() / C;
    int stride_C = input.stride(1);
    int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);
    auto norms = torch::zeros({total_vectors}, input.options());

    const int threads = 256;
    int blocksPerVector = (C + threads - 1) / threads;
    dim3 grid(total_vectors, blocksPerVector);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_reduce_refactored", ([&] {
        l2_norm_reduce_kernel_refactored<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            norms.data_ptr<scalar_t>(),
            C,
            stride_C,
            outer_stride
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_normalize_refactored", ([&] {
        l2_norm_normalize_kernel_refactored<scalar_t><<<grid, threads>>>(
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
    m.def("forward", &forward, "Refactored Modular L2 normalization with device functions");
}
