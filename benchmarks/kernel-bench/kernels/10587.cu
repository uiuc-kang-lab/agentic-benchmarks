#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void cumprod_min_sync_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_chains) {

    int chain_id = blockIdx.x;
    if (chain_id >= total_chains) return;

    int batch_idx = chain_id / stride;
    int in_idx = chain_id % stride;
    int64_t base = batch_idx * (dim_size * stride) + in_idx;

    int t = threadIdx.x;
    int T = blockDim.x;
    int chunk = (dim_size + T - 1) / T;
    
    __shared__ scalar_t s_local[512];

    // Calculate segment bounds
    int start = t * chunk;
    int end = min(start + chunk, dim_size);

    // First pass: compute local products without synchronization
    scalar_t local_prod = 1;
    #pragma unroll
    for (int i = start; i < end; i++) {
        int64_t idx = base + i * stride;
        local_prod *= input[idx];
    }
    
    // Store local product and synchronize once
    s_local[t] = local_prod;
    __syncthreads();  // First necessary sync: ensure all local products are visible

    // Compute exclusive prefix product
    scalar_t offset = 1;
    #pragma unroll
    for (int i = 0; i < t; i++) {
        offset *= s_local[i];
    }
    __syncthreads();  // Second necessary sync: ensure all threads have correct offsets

    // Final pass: compute and write results without further synchronization
    scalar_t running_prod = offset;
    #pragma unroll
    for (int i = start; i < end; i++) {
        int64_t idx = base + i * stride;
        running_prod *= input[idx];
        output[idx] = running_prod;
    }
}

torch::Tensor cumprod_min_sync_forward(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    auto sizes = input.sizes();
    auto strides = input.strides();
    int64_t dim_size = sizes[dim];
    int64_t stride_val = strides[dim];
    int64_t total_chains = input.numel() / dim_size;
    
    const int threads = 512;
    dim3 blocks(total_chains);
    dim3 threads_per_block(threads);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_min_sync", ([&] {
        cumprod_min_sync_kernel<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &cumprod_min_sync_forward, "Cumulative product forward with minimal synchronization (CUDA)");
}