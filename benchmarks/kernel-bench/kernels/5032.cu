#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel performs a partial reduction of the squared elements for each vector using a 2D grid.
// Each block (indexed by blockIdx.y) computes a partial sum and then uses an atomicAdd (once per block) to update
// the global accumulator for the vector. This minimizes global atomic calls to one per block.

template <typename scalar_t>
__global__ void l2_norm_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ norms,
    const int C,
    const int stride_C,
    const int outer_stride) {

    const int vec_idx = blockIdx.x;  // Each vector
    const int grid_y = gridDim.y;    // Number of blocks working on one vector
    const int base_offset = vec_idx * outer_stride;

    scalar_t partial_sum = 0;
    // Each block processes a subset of the C dimension
    int start = blockIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * grid_y;
    for (int i = start; i < C; i += stride) {
        scalar_t val = input[base_offset + i * stride_C];
        partial_sum += val * val;
    }

    // Warp-level reduction using shuffle
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }

    // Use shared memory to further reduce the sums from different warps
    __shared__ scalar_t shared[32]; // Enough to hold one value per warp
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) {
        shared[warp_id] = partial_sum;
    }
    __syncthreads();

    // The first thread of the block reduces the warp sums and updates the global result with one atomicAdd call
    if (threadIdx.x == 0) {
        scalar_t block_sum = 0;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int i = 0; i < num_warps; i++) {
            block_sum += shared[i];
        }
        atomicAdd(&norms[vec_idx], block_sum);
    }
}

// This kernel normalizes the input vector using the computed norm from the reduction phase.
// It uses the same 2D grid configuration to cover the C dimension of each vector.

template <typename scalar_t>
__global__ void l2_norm_normalize_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ norms,
    const int C,
    const int stride_C,
    const int outer_stride) {

    const int vec_idx = blockIdx.x;
    const int base_offset = vec_idx * outer_stride;

    // Read the squared norm and compute its square root with an epsilon for stability
    scalar_t norm_val = sqrt(norms[vec_idx]) + static_cast<scalar_t>(1e-12);
    scalar_t inv_norm = static_cast<scalar_t>(1.0) / norm_val;

    int grid_y = gridDim.y;
    int start = blockIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * grid_y;
    for (int i = start; i < C; i += stride) {
        int index = base_offset + i * stride_C;
        output[index] = input[index] * inv_norm;
    }
}

// The forward function launches two kernels: one for reduction and one for normalization.
// A temporary tensor 'norms' is allocated to store the squared sums, and is zero-initialized.

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");

    // Normalization along dimension 1
    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0); // Assumes vectors are contiguous in dim 1

    auto output = torch::empty_like(input);
    auto norms = torch::zeros({total_vectors}, input.options());

    // Determine grid configuration
    const int threads = 256;
    int blocksPerVector = (C + threads - 1) / threads;  // Number of blocks (in y-dimension) per vector
    dim3 grid(total_vectors, (blocksPerVector + 1) / 2);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_reduce_atomic", ([&] {
        l2_norm_reduce_kernel<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            norms.data_ptr<scalar_t>(),
            C,
            stride_C,
            outer_stride
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_normalize_atomic", ([&] {
        l2_norm_normalize_kernel<scalar_t><<<grid, threads>>>(
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
    m.def("forward", &forward, "L2 normalization with reduced global atomics via two-phase kernel");
}
