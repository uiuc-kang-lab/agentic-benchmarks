#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t, unsigned int BLOCK_SIZE = 256>
__global__ void subtract_shifted_cumsum_shared(const scalar_t* __restrict__ input, 
                                             scalar_t* __restrict__ output,
                                             const int64_t dim_size,
                                             const int64_t num_elements) {
    __shared__ scalar_t shared_data[BLOCK_SIZE];
    
    const int64_t tid = threadIdx.x;
    const int64_t gid = blockIdx.x * BLOCK_SIZE + tid;
    const int64_t slice = gid / dim_size;
    
    if (slice * dim_size >= num_elements) return;
    
    // Load data into shared memory
    if (gid < num_elements && tid < dim_size) {
        shared_data[tid] = input[slice * dim_size + tid];
    }
    __syncthreads();
    
    if (tid < dim_size && gid < num_elements) {
        const scalar_t total = shared_data[dim_size - 1];
        
        if (tid == 0) {
            output[slice * dim_size] = total;
        } else {
            output[slice * dim_size + tid] = total - shared_data[tid - 1];
        }
    }
}

at::Tensor reverse_cumsum_optimized(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    
    const int64_t dim_size = x.size(dim);
    const int64_t num_elements = x.numel();
    const int64_t num_slices = num_elements / dim_size;
    
    auto cumsum = x.cumsum(dim);
    auto output = torch::empty_like(x);
    
    constexpr int BLOCK_SIZE = 256;
    const int blocks = (num_slices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    AT_DISPATCH_ALL_TYPES(x.scalar_type(), "reverse_cumsum", [&] {
        subtract_shifted_cumsum_shared<scalar_t, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
            cumsum.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            num_elements
        );
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum_optimized, "Optimized reverse cumsum with shared memory");
}