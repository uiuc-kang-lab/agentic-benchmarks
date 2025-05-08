#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel performs L2 norm reduction using explicit stride loops to handle large workloads
template <typename scalar_t>
__global__ void l2_norm_reduce_stride_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ norms,
    const int C,
    const int stride_C,
    const int outer_stride) {

    // Each block processes one vector in the x-dimension; multiple blocks (gridDim.y) handle the vector's C elements
    int vec_idx = blockIdx.x;  // Index of the vector
    int base = vec_idx * outer_stride;

    // Calculate total number of threads working per vector
    int total_threads = gridDim.y * blockDim.x;
    // Compute a unique thread id within the vector's workload using the y-dimension of the grid
    int thread_index = threadIdx.x + blockIdx.y * blockDim.x;

    scalar_t sum = 0;
    // Stride loop: each thread processes multiple elements if C is larger than total_threads
    for (int i = thread_index; i < C; i += total_threads) {
        // Boundary is enforced by loop condition (i < C)
        scalar_t val = input[base + i * stride_C];
        sum += val * val;
    }

    // Perform warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Use shared memory for block-level reduction
    __shared__ scalar_t shared[32];
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction: first thread of the block accumulates results from all warps and atomically updates the global norm sum
    if (threadIdx.x == 0) {
        scalar_t block_sum = 0;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int i = 0; i < num_warps; i++) {
            block_sum += shared[i];
        }
        atomicAdd(&norms[vec_idx], block_sum);
    }
}

// This kernel normalizes each vector using the L2 norm computed in the reduction kernel
template <typename scalar_t>
__global__ void l2_norm_normalize_stride_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ norms,
    const int C,
    const int stride_C,
    const int outer_stride) {

    int vec_idx = blockIdx.x;
    int base = vec_idx * outer_stride;

    // Compute the inverse norm with an epsilon for numerical stability
    scalar_t norm_val = sqrt(norms[vec_idx]) + static_cast<scalar_t>(1e-12);
    scalar_t inv_norm = static_cast<scalar_t>(1.0) / norm_val;

    int total_threads = gridDim.y * blockDim.x;
    int thread_index = threadIdx.x + blockIdx.y * blockDim.x;

    // Stride loop to normalize each element, ensuring boundary conditions via the loop condition
    for (int i = thread_index; i < C; i += total_threads) {
        int index = base + i * stride_C;
        output[index] = input[index] * inv_norm;
    }
}

// Forward function to launch kernels
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0); // This assumes the vectors are stored contiguously along dim 1

    auto output = torch::empty_like(input);
    auto norms = torch::zeros({total_vectors}, input.options());

    // Configure the grid: x-dimension for vectors, y-dimension for splitting the C-dimension workload
    const int threads = 256;
    int blocksPerVector = (C + threads - 1) / threads; // Number of blocks needed per vector
    dim3 grid(total_vectors, blocksPerVector);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_reduce_stride", ([&] {
        l2_norm_reduce_stride_kernel<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            norms.data_ptr<scalar_t>(),
            C,
            stride_C,
            outer_stride
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_normalize_stride", ([&] {
        l2_norm_normalize_stride_kernel<scalar_t><<<grid, threads>>>(
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
    m.def("forward", &forward, "L2 normalization using stride loops for large workloads");
}
