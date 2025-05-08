#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses warp shuffle intrinsics to perform reduction with uniform control flow
// Each block computes one output element corresponding to a unique (outer, inner) index

template <typename scalar_t>
__global__ void sum_reduce_warp_shuffle_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size) {

    // Each block is responsible for one output element
    int out_idx = blockIdx.x; // flattened index over outer * inner
    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;
    int64_t base = outer_idx * reduce_size * inner_size + inner_idx;

    // Each thread accumulates a partial sum over the reduction dimension
    scalar_t sum = 0;
    for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        sum += input[base + i * inner_size];
    }

    // Intra-warp reduction using warp shuffle to avoid divergent branches
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Use shared memory to accumulate results from different warps
    __shared__ scalar_t warp_sum[32]; // Enough space for up to 32 warps per block
    int warp_id = threadIdx.x / warpSize;
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        warp_sum[warp_id] = sum;
    }
    __syncthreads();

    // First warp now reduces the partial sums from all warps
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < num_warps) {
        sum = warp_sum[threadIdx.x];
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        if (threadIdx.x == 0) {
            output[out_idx] = sum;
        }
    }
}

// Host function that prepares tensor dimensions and launches the warp-shuffle reduction kernel

torch::Tensor sum_reduce_warp_shuffle_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    // Compute sizes: outer, reduce, and inner
    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Prepare output tensor shape (with dimension 'dim' collapsed to 1)
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Launch kernel: one block per output element
    int64_t num_output = outer_size * inner_size;
    int threads = 256;
    int blocks = num_output;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_warp_shuffle_cuda", ([&] {
        sum_reduce_warp_shuffle_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_warp_shuffle_cuda, "Sum reduction with warp shuffle to minimize divergence (CUDA)");
}
