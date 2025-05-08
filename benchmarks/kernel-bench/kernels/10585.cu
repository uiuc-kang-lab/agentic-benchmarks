#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel computes the cumulative product along a contiguous dimension with minimal synchronizations.
// Each block processes one cumulative product chain (row). The work is divided among threads, where each thread
// computes the product of its assigned segment, writes its local product to shared memory, then, after a single
// __syncthreads(), computes an exclusive offset (the product of local products of threads with lower indices).
// Finally, each thread re-reads its assigned segment and writes the cumulative product to global memory.
// Only one synchronization call is used to ensure shared memory consistency, reducing overhead.

template <typename scalar_t>
__global__ void cumprod_min_sync_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_chains) {

    // Each block processes one cumulative product chain
    int chain_id = blockIdx.x;
    if (chain_id >= total_chains) return;

    int batch_idx = chain_id / stride;
    int in_idx = chain_id % stride;
    int64_t base = batch_idx * (dim_size * stride) + in_idx;

    int t = threadIdx.x;
    int T = blockDim.x;
    int chunk = (dim_size + T - 1) / T;
    int start = t * chunk;
    int end = start + chunk;
    if (end > dim_size) end = dim_size;

    // Allocate shared memory for local products; only one __syncthreads() is needed here.
    extern __shared__ char shared_mem[];
    scalar_t* s_local = reinterpret_cast<scalar_t*>(shared_mem);

    // First pass: each thread computes the product over its assigned segment
    scalar_t local_prod = 1;
    for (int i = start; i < end; i++) {
        int64_t idx = base + i * stride;
        local_prod *= input[idx];
    }
    s_local[t] = local_prod;
    __syncthreads();  // Synchronize to ensure all local products are written

    // Compute exclusive prefix product offset for this thread
    scalar_t offset = 1;
    for (int i = 0; i < t; i++) {
        offset *= s_local[i];
    }

    // Second pass: compute the cumulative product for the assigned segment using the offset
    scalar_t prod = offset;
    for (int i = start; i < end; i++) {
        int64_t idx = base + i * stride;
        prod *= input[idx];
        output[idx] = prod;
    }
}

// CUDA forward function: assumes the cumulative product is along a contiguous dimension.
// The kernel launches one block per cumulative product chain.

torch::Tensor cumprod_cuda_min_sync_forward(torch::Tensor input, int64_t dim) {
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    auto output = torch::empty_like(input);

    auto sizes = input.sizes();
    auto strides = input.strides();
    int64_t dim_size = sizes[dim];
    int64_t stride_val = strides[dim];
    int64_t total_chains = input.numel() / dim_size;

    // Use 512 threads per block; adjust shared memory size accordingly
    int threads = 512;
    dim3 blocks(total_chains);
    dim3 threads_per_block(threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_min_sync", ([&] {
        cumprod_min_sync_kernel<scalar_t><<<blocks, threads_per_block, threads * sizeof(scalar_t)>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            dim_size,
            stride_val,
            total_chains
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda_min_sync_forward, "Minimal synchronization cumulative product forward (CUDA)");
}
