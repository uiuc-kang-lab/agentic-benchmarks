#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Device function: Computes the product of a segment of the cumulative product chain
// from index 'start' to 'end' (non-inclusive) using the given stride.

template <typename scalar_t>
__device__ inline scalar_t compute_segment_product(const scalar_t* __restrict__ input, int64_t base, int start, int end, int64_t stride) {
    scalar_t prod = 1;
    for (int i = start; i < end; ++i) {
        prod *= input[base + i * stride];
    }
    return prod;
}

// Device function: Writes the cumulative product for a segment starting from a given offset.
// It recomputes the cumulative product for the segment and writes the results to output.

template <typename scalar_t>
__device__ inline void write_segment_cumprod(scalar_t offset, const scalar_t* __restrict__ input, scalar_t* __restrict__ output, int64_t base, int start, int end, int64_t stride) {
    scalar_t prod = offset;
    for (int i = start; i < end; ++i) {
        prod *= input[base + i * stride];
        output[base + i * stride] = prod;
    }
}

// Device function: Computes the exclusive prefix product from the shared array sdata up to index tid.
// Each thread's offset is the product of all previous threads' local segment products.

template <typename scalar_t>
__device__ inline scalar_t compute_exclusive_prefix(const scalar_t* __restrict__ sdata, int tid) {
    scalar_t offset = 1;
    for (int i = 0; i < tid; ++i) {
        offset *= sdata[i];
    }
    return offset;
}

// Modular two-pass kernel: each block processes one cumulative product chain (row) and splits it among threads.
// This kernel uses helper functions for computing local segment products and then writing the final cumulative product.

template <typename scalar_t>
__global__ void modular_parallel_cumprod_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_chains) {

    int chain_id = blockIdx.x;  // one block per chain
    if (chain_id >= total_chains) return;

    int batch_idx = chain_id / stride;
    int in_idx = chain_id % stride;
    int64_t base = batch_idx * (dim_size * stride) + in_idx;

    int tid = threadIdx.x;
    int T = blockDim.x;
    int chunk = (dim_size + T - 1) / T;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > dim_size) end = dim_size;

    // Shared memory to store each thread's local product
    extern __shared__ char shared[];
    scalar_t* s_local = reinterpret_cast<scalar_t*>(shared);

    // First pass: Each thread computes the product of its assigned segment
    scalar_t local_prod = compute_segment_product<scalar_t>(input, base, start, end, stride);
    s_local[tid] = local_prod;
    __syncthreads();

    // Compute the exclusive prefix product offset for this thread
    scalar_t offset = compute_exclusive_prefix<scalar_t>(s_local, tid);
    __syncthreads();

    // Second pass: Re-read input and write the final cumulative product for this segment
    write_segment_cumprod<scalar_t>(offset, input, output, base, start, end, stride);
}

// CUDA forward function that launches one block per cumulative product chain
// The cumulative product is performed along the specified dimension, which is assumed to be contiguous.

torch::Tensor modular_cumprod_cuda_forward(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    auto sizes = input.sizes();
    auto strides = input.strides();
    int64_t dim_size = sizes[dim];
    int64_t stride_val = strides[dim];
    int64_t total_chains = input.numel() / dim_size;
    
    const int threads = 256;
    dim3 blocks(total_chains);
    dim3 threads_per_block(threads);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "modular_cumprod_cuda", ([&] {
        modular_parallel_cumprod_kernel<scalar_t><<<blocks, threads_per_block, threads * sizeof(scalar_t)>>>(
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
    m.def("forward", &modular_cumprod_cuda_forward, "Modular Parallel cumulative product forward (CUDA)");
}
