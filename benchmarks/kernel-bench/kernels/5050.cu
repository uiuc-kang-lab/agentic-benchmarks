#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Structure to hold constant parameters for L2 normalization
struct L2NormConstants {
    int C;          // number of elements per vector
    int stride_C;   // stride between elements in a vector
    int outer_stride; // stride between vectors
};

// Declare constant memory for frequently accessed parameters
__constant__ L2NormConstants l2norm_constants;

// Device function: Warp-level reduction
template <typename scalar_t>
__device__ inline scalar_t warpReduceSum(scalar_t val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Device function: Block-level reduction using shared memory
template <typename scalar_t>
__device__ inline scalar_t blockReduceSum(scalar_t val) {
    __shared__ scalar_t shared[32]; // one element per warp
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < num_warps) ? shared[lane] : static_cast<scalar_t>(0);
    if (warp_id == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// Kernel: Reduction phase for L2 norm computation
// Reads constant parameters from __constant__ memory
template <typename scalar_t>
__global__ void l2_norm_reduce_kernel_const(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ norms) {

    int C = l2norm_constants.C;
    int stride_C = l2norm_constants.stride_C;
    int outer_stride = l2norm_constants.outer_stride;

    int vec_idx = blockIdx.x; // Each blockIdx.x corresponds to one vector
    int base_offset = vec_idx * outer_stride;
    int grid_y = gridDim.y; // Number of blocks along the C dimension

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

// Kernel: Normalization phase for L2 norm computation
// Uses constant parameters from __constant__ memory
template <typename scalar_t>
__global__ void l2_norm_normalize_kernel_const(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ norms) {

    int C = l2norm_constants.C;
    int stride_C = l2norm_constants.stride_C;
    int outer_stride = l2norm_constants.outer_stride;

    int vec_idx = blockIdx.x;
    int base_offset = vec_idx * outer_stride;

    // Compute the L2 norm with stability epsilon
    scalar_t norm_val = sqrt(norms[vec_idx]) + static_cast<scalar_t>(1e-12);
    scalar_t inv_norm = static_cast<scalar_t>(1.0) / norm_val;

    int grid_y = gridDim.y;
    int start = threadIdx.x + blockIdx.y * blockDim.x;
    int stride = blockDim.x * grid_y;

    // Normalize the vector elements
    for (int i = start; i < C; i += stride) {
        int index = base_offset + i * stride_C;
        output[index] = input[index] * inv_norm;
    }
}

// Forward function: Sets up constant memory and launches the two-phase kernels
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");

    int C = input.size(1);
    int total_vectors = input.numel() / C;
    int stride_C = input.stride(1);
    int outer_stride = input.stride(0); // Assumes vectors are contiguous along dim 1

    // Copy read-only parameters to constant memory
    L2NormConstants h_const;
    h_const.C = C;
    h_const.stride_C = stride_C;
    h_const.outer_stride = outer_stride;
    cudaMemcpyToSymbol(l2norm_constants, &h_const, sizeof(L2NormConstants));

    auto output = torch::empty_like(input);
    auto norms = torch::zeros({total_vectors}, input.options());

    const int threads = 256;
    int blocksPerVector = (C + threads - 1) / threads;
    dim3 grid(total_vectors, blocksPerVector);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_reduce_const", ([&] {
        l2_norm_reduce_kernel_const<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            norms.data_ptr<scalar_t>()
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_normalize_const", ([&] {
        l2_norm_normalize_kernel_const<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            norms.data_ptr<scalar_t>()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization with constant memory for frequently accessed parameters");
}
