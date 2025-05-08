#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Improved kernel reducing atomics to minimize global memory contention.
template <typename scalar_t>
__global__ void cumprod_atomic_minimized_kernel(
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

    int t = threadIdx.x;
    int T = blockDim.x;
    int chunk = (dim_size + T - 1) / T;

    extern __shared__ scalar_t s_local[];

    int start = t * chunk;
    int end = start + chunk;
    if (end > dim_size) end = dim_size;

    scalar_t local_prod = 1;
    for (int i = start; i < end; i++) {
        int64_t idx = base + i * stride;
        local_prod *= input[idx];
    }
    s_local[t] = local_prod;
    __syncthreads();

    if (t == 0) {
        scalar_t total_prod = 1;
        for (int i = 0; i < T; i++) {
            total_prod *= s_local[i];
            s_local[i] = total_prod;
        }
    }

    __syncthreads();

    scalar_t offset = (t == 0) ? 1 : s_local[t - 1];

    scalar_t prod = offset;
    for (int i = start; i < end; i++) {
        int64_t idx = base + i * stride;
        prod *= input[idx];
        output[idx] = prod;
    }
}

// Forward function for atomic minimized kernel

torch::Tensor cumprod_cuda_atomic_minimized_forward(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
  
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    auto sizes = input.sizes();
    int64_t dim_size = sizes[dim];
    int64_t stride_val = input.strides()[dim];
    int64_t total_chains = input.numel() / dim_size;
    
    int threads = 256;
    dim3 blocks(total_chains);
    dim3 threads_per_block(threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_atomic_minimized", ([&] {
        cumprod_atomic_minimized_kernel<scalar_t><<<blocks, threads_per_block, threads * sizeof(scalar_t)>>>(
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
    m.def("forward", &cumprod_cuda_atomic_minimized_forward, "Optimized cumulative product forward with minimized atomics (CUDA)");
}