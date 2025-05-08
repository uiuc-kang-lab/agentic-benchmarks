#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel that uses warp-level primitives and shared memory for intra-block reduction
// to perform cumulative product operations.
// Each block processes one cumulative product chain (row) along a contiguous dimension.

template <typename scalar_t>
__global__ void cumprod_warp_reduce_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_chains) {

    int chain_id = blockIdx.x; // one block per chain
    if (chain_id >= total_chains) return;

    // Decode chain index into batch and in-dimension index
    int batch_idx = chain_id / stride;
    int in_idx = chain_id % stride;
    int64_t base = batch_idx * (dim_size * stride) + in_idx;

    int t = threadIdx.x;
    int T = blockDim.x; 
    // Divide the chain into chunks for each thread
    int chunk = (dim_size + T - 1) / T;
    int start = t * chunk;
    int end = start + chunk;
    if (end > dim_size) end = dim_size;

    // Each thread computes the product over its chunk
    scalar_t local_prod = 1;
    for (int i = start; i < end; i++) {
        int64_t idx = base + i * stride;
        local_prod *= input[idx];
    }

    // Perform warp-level exclusive scan of local_prod using warp-level primitives
    unsigned fullMask = 0xffffffff;
    int lane = t & 31;         // thread index within the warp
    int warp_id = t >> 5;      // warp index within the block

    // Compute inclusive scan within the warp
    scalar_t inclusive = local_prod;
    for (int offset = 1; offset < 32; offset *= 2) {
        scalar_t tmp = __shfl_up_sync(fullMask, inclusive, offset);
        if (lane >= offset)
            inclusive = tmp * inclusive;
    }
    // Convert inclusive scan to exclusive scan by shifting right
    scalar_t warp_exclusive = (lane == 0) ? scalar_t(1) : __shfl_sync(fullMask, inclusive, lane - 1);

    // Store the total product of each warp (i.e. the inclusive value from the last lane) in shared memory
    __shared__ scalar_t warp_totals[32]; // assuming blockDim.x <= 1024
    if (lane == 31) {
        warp_totals[warp_id] = inclusive;
    }
    __syncthreads();

    // Each warp computes an offset by multiplying the totals of all preceding warps
    scalar_t warp_offset = 1;
    if (warp_id > 0) {
        for (int i = 0; i < warp_id; i++) {
            warp_offset *= warp_totals[i];
        }
    }
    __syncthreads();

    // Final exclusive offset for this thread
    scalar_t thread_offset = warp_offset * warp_exclusive;

    // Second pass: Recompute cumulative product for the thread's chunk starting from its offset
    scalar_t prod = thread_offset;
    for (int i = start; i < end; i++) {
        int64_t idx = base + i * stride;
        prod *= input[idx];
        output[idx] = prod;
    }
}

// Forward function launching the kernel. Assumes the cumulative product is computed along a contiguous dimension.

torch::Tensor cumprod_cuda_warp_reduce_forward(torch::Tensor input, int64_t dim) {
    // Ensure the input tensor is contiguous along the dimension of interest
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    auto sizes = input.sizes();
    auto strides = input.strides();
    int64_t dim_size = sizes[dim];
    int64_t stride_val = strides[dim];
    int64_t total_chains = input.numel() / dim_size; // number of independent cumprod sequences

    auto output = torch::empty_like(input);

    // Launch one block per chain; 256 threads per block for example
    int threads = 256;
    int blocks = total_chains;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_warp_reduce", ([&] {
        cumprod_warp_reduce_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &cumprod_cuda_warp_reduce_forward, "Cumulative product forward with warp-level reduction (CUDA)");
}
