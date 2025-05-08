#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel uses warp-level primitives to perform the cumulative product operation.
// Each warp processes a segment of the cumulative product chain, and warp-level shuffles
// are used to compute the prefix product efficiently within the warp.

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

// Warp-level cumulative product kernel
// Each warp processes a segment of the cumulative product chain.
// Warp-level shuffles are used to compute the prefix product efficiently within the warp.

template <typename scalar_t>
__global__ void warp_level_cumprod_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_chains) {

    int chain_id = blockIdx.x;  // one block per chain
    if (chain_id >= total_chains) return;

    // Decode the chain to get batch index and in-dimension index
    int batch_idx = chain_id / stride;
    int in_idx = chain_id % stride;
    int64_t base = batch_idx * (dim_size * stride) + in_idx;

    int t = threadIdx.x;
    int warp_id = t / 32;
    int lane_id = t % 32;
    int T = blockDim.x;
    int chunk = (dim_size + T - 1) / T;

    // Each warp processes a segment of the chain
    int start = warp_id * chunk;
    int end = start + chunk;
    if (end > dim_size) end = dim_size;

    // Compute the product of the segment assigned to this warp
    scalar_t local_prod = compute_segment_product<scalar_t>(input, base, start, end, stride);

    // Use warp-level shuffle to compute the prefix product within the warp
    for (int offset = 1; offset < 32; offset *= 2) {
        scalar_t val = __shfl_up_sync(0xFFFFFFFF, local_prod, offset);
        if (lane_id >= offset) local_prod *= val;
    }

    // Write the cumulative product for the segment
    if (lane_id == 31) {
        output[base + end * stride - 1] = local_prod;
    }

    __syncthreads();

    // Each thread writes its result back to global memory
    for (int i = start + lane_id; i < end; i += 32) {
        output[base + i * stride] = local_prod;
    }
}

// CUDA forward function that launches one block per cumulative product chain

torch::Tensor warp_level_cumprod_cuda_forward(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    // Extract tensor info
    auto sizes = input.sizes();
    auto strides = input.strides();
    int64_t dim_size = sizes[dim];
    int64_t stride_val = strides[dim];
    // total number of independent cumulative product chains
    int64_t total_chains = input.numel() / dim_size;
    
    // Launch one block per chain. We choose 256 threads per block to distribute the workload evenly along the chain.
    int threads = 256;
    dim3 blocks(total_chains);
    dim3 threads_per_block(threads);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "warp_level_cumprod_cuda", ([&] {
        warp_level_cumprod_kernel<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &warp_level_cumprod_cuda_forward, "Warp-level cumulative product forward (CUDA)");
}
