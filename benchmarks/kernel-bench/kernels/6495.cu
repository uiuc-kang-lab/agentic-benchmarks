#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel that uses warp-level primitives for final reduction
template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {

    // Allocate shared memory for storing per-warp sums
    extern __shared__ char shared_mem[];
    // Number of warps per block = blockDim.x / 32
    scalar_t* warp_sums = reinterpret_cast<scalar_t*>(shared_mem);

    // Identify thread and warp indices
    const int tid = threadIdx.x;
    const int lane = tid & 31;  // thread index within warp
    const int warpId = tid >> 5;  // warp index

    // Each block computes one output element
    const int output_idx = blockIdx.x;
    if (output_idx >= outer_size * inner_size)
        return;

    // Compute corresponding indices for the reduction dimension
    const int outer_idx = output_idx / inner_size;
    const int inner_idx = output_idx % inner_size;
    const int base_idx = outer_idx * dim_size * inner_size + inner_idx;

    // Each thread computes a partial sum over the reduction dimension
    scalar_t sum = 0;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        sum += input[base_idx + i * inner_size];
    }

    // Use warp-level reduction with __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Write the reduced value of each warp to shared memory
    if (lane == 0) {
        // Divide by dim_size only once per warp instead of at the very end
        warp_sums[warpId] = sum / static_cast<scalar_t>(dim_size);
    }
    __syncthreads();

    // Let the first warp reduce the warp sums
    if (tid < (blockDim.x / 32)) {
        sum = warp_sums[tid];
        // Number of warp sums is (blockDim.x / 32); perform warp reduction
        for (int offset = (blockDim.x / 32) / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            // Write the final sum (already divided by dim_size in each warp)
            output[output_idx] = sum;
        }
    }
}

// Host function to prepare and launch the CUDA kernel
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    // Compute sizes
    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Remove the reduced dimension for output shape
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    const int threads = 256;
    const int blocks = outer_size * inner_size;
    // Allocate shared memory: one scalar per warp
    const int shared_mem_size = (threads / 32) * input.element_size();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &mean_reduce_cuda, "Mean reduction (CUDA)");
}
