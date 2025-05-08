#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel distributes the cumulative product workloads evenly by processing multiple chains concurrently
// within a single block using a 2D block layout. Each chain is partitioned among the threads in the x-direction,
// and each chain is assigned a row in the block (y-direction). Each thread computes the product of its segment,
// then a simple exclusive prefix (offset) is computed per chain so that each segment multiplies correctly in order.
// This design maximizes thread utilization and reduces bottlenecks for chains of various lengths.

template <typename scalar_t>
__global__ void even_parallel_cumprod_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t L,        // length of each cumulative product chain (dim_size)
    const int64_t stride,   // stride between consecutive elements in the chain
    const int64_t total_chains) {

    // 2D block layout: threadIdx.y indexes chains within the block, threadIdx.x works on segments of each chain
    int local_chain = threadIdx.y;                  // Chain index within the block
    int global_chain = blockIdx.x * blockDim.y + local_chain; // Global chain index
    if (global_chain >= total_chains) return;

    int t = threadIdx.x;
    int T = blockDim.x;
    int chunk = (L + T - 1) / T; // number of elements per thread in the chain

    // Compute the base index for the chain from global chain id
    // Decoding similar to previous implementations: chain id = batch_idx * stride + in_idx
    int batch_idx = global_chain / stride;
    int in_idx = global_chain % stride;
    int64_t base = batch_idx * (L * stride) + in_idx;

    // Shared memory layout: each chain gets T elements (stored contiguously)
    extern __shared__ char shared_mem[];
    scalar_t* s_local = reinterpret_cast<scalar_t*>(shared_mem);
    // For each chain, allocate a segment of size T in shared memory
    scalar_t* chain_partial = s_local + local_chain * T;

    // First pass: Each thread computes the product of its assigned segment sequentially
    scalar_t local_prod = 1;
    int start = t * chunk;
    int end = start + chunk;
    if (end > L) end = L;
    for (int i = start; i < end; i++) {
        int64_t idx = base + i * stride;
        local_prod = local_prod * input[idx];
    }
    chain_partial[t] = local_prod;
    __syncthreads();

    // Compute exclusive prefix (offset) for this thread within the chain
    scalar_t offset = 1;
    for (int i = 0; i < t; i++) {
        offset = offset * chain_partial[i];
    }
    __syncthreads();

    // Second pass: Recompute the cumulative product for the assigned segment starting with the offset
    scalar_t prod = offset;
    for (int i = start; i < end; i++) {
        int64_t idx = base + i * stride;
        prod = prod * input[idx];
        output[idx] = prod;
    }
}

// CUDA forward function
// The cumulative product is applied along the dimension specified by 'dim', which is assumed to be contiguous.
// The tensor is interpreted as containing "total_chains" of length L, where L = sizes[dim] and stride = strides[dim].

torch::Tensor even_parallel_cumprod_forward(torch::Tensor input, int64_t dim) {
    // Ensure the input tensor is contiguous
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    auto sizes = input.sizes();
    auto strides = input.strides();
    int64_t L = sizes[dim];           // Length of the cumulative product dimension
    int64_t stride_val = strides[dim];  // Stride for that dimension
    int64_t total_chains = input.numel() / L;  // Total independent cumulative product chains

    auto output = torch::empty_like(input);

    // Define 2D block configuration
    // For example, use 256 threads in x-direction and 8 chains per block in y-direction
    const int threads_x = 256;
    const int threads_y = 8;
    dim3 threads_per_block(threads_x, threads_y);

    // Determine number of blocks needed: each block covers 'threads_y' chains
    int blocks_x = (total_chains + threads_y - 1) / threads_y;
    dim3 blocks(blocks_x);

    // Calculate shared memory size = (threads_x * threads_y * sizeof(scalar_t))
    size_t shared_memory_size = threads_x * threads_y * sizeof(torch::ScalarType::Float /*dummy*/);
    // It's safer to compute shared_memory_size in the dispatch lambda using sizeof(scalar_t)

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "even_parallel_cumprod", ([&] {
        even_parallel_cumprod_kernel<scalar_t><<<blocks, threads_per_block, threads_x * threads_y * sizeof(scalar_t)>>> (
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            L,
            stride_val,
            total_chains
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &even_parallel_cumprod_forward, "Evenly distributed parallel cumulative product forward (CUDA)");
}
