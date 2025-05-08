#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

#define BLOCK_SIZE 256

// Kernel for reduction with warp-coalesced memory access
// Each thread block computes a partial sum for a slice of the reduction dimension
// and writes the result to the output tensor.
template <typename scalar_t>
__global__ void mean_reduce_kernel_warp_coalesced(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;

    int outer_idx = tid / inner_size;
    int inner_idx = tid % inner_size;
    int64_t offset = outer_idx * (dim_size * inner_size) + inner_idx;

    scalar_t sum = static_cast<scalar_t>(0);

    // Each thread processes elements in a strided manner to ensure coalesced memory access
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        sum += __ldg(input + offset + i * inner_size);
    }

    // Reduce within the block using shared memory
    __shared__ scalar_t sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Perform parallel reduction within the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write the result for this block to output
    if (threadIdx.x == 0) {
        output[tid] = sdata[0] / static_cast<scalar_t>(dim_size);
    }
}

// Host function launching the kernel
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    const int threads = BLOCK_SIZE;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel_warp_coalesced<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Mean reduction with warp-coalesced memory access (CUDA)");
}