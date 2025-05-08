#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel that divides the workload using strided loops to handle large data segments
// Each thread performs its assigned work in a strided manner, which ensures all elements are processed correctly

// This kernel processes one cumulative product chain per block and uses a strided approach within each thread
// to incrementally accumulate the product over time. Threads wrap around the array using strided loops
// to ensure that we handle arrays larger than the number of logical threads per block

template <typename scalar_t>
__global__ void strided_cumprod_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_chains) {

    for(int chain_id = blockIdx.x; chain_id < total_chains; chain_id += gridDim.x) {
        int batch_idx = chain_id / stride;
        int in_idx = chain_id % stride;
        int64_t base = batch_idx * (dim_size * stride) + in_idx;

        int t = threadIdx.x;

        scalar_t product = 1;
        for(int idx = t; idx < dim_size; idx += blockDim.x) {
            int64_t gid = base + idx * stride;
            product *= input[gid];
            output[gid] = product;
        }
    }
}

// Forward function using strided workload approach

torch::Tensor cumprod_strided_cuda_forward(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    // Extract tensor info
    auto sizes = input.sizes();
    auto strides = input.strides();
    int64_t dim_size = sizes[dim];
    int64_t stride_val = strides[dim];
    int64_t total_chains = input.numel() / dim_size;
    
    // Setup kernel parameters
    const int threads = 256;
    int blocks = std::min(total_chains, static_cast<int64_t>((input.numel() + threads - 1) / threads));
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_strided_cuda", ([&] {
        strided_cumprod_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &cumprod_strided_cuda_forward, "Strided cumulative product forward (CUDA)");
}