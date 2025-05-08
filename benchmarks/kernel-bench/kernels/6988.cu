#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

__constant__ int const_mem_k[1];

template <typename scalar_t>
__global__ void optimized_argmin_kernel(const scalar_t* __restrict__ x,
                                      int64_t* __restrict__ output,
                                      int64_t outer_size,
                                      int64_t inner_size) {
    extern __shared__ char shared_mem[];
    scalar_t* shared_vals = reinterpret_cast<scalar_t*>(shared_mem);
    int* shared_indices = reinterpret_cast<int*>(shared_vals + blockDim.x);
    
    const int K = const_mem_k[0];
    const int tid = threadIdx.x;
    const int outer = blockIdx.y;
    const int inner_block = blockIdx.x;
    const int inner = inner_block * blockDim.x + tid;
    
    if(inner >= inner_size) return;
    
    const scalar_t* slice_start = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;
    scalar_t min_val = slice_start[0];
    int min_index = 0;
    
    #pragma unroll 4
    for(int k = 1; k < K; k++) {
        scalar_t val = slice_start[k * inner_size];
        if(val < min_val) {
            min_val = val;
            min_index = k;
        }
    }
    
    output[outer * inner_size + inner] = min_index;
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    int dims = x.dim();
    dim = dim < 0 ? dim + dims : dim;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");
    
    int64_t outer_size = 1;
    for(int i = 0; i < dim; i++) outer_size *= x.size(i);
    
    int K = static_cast<int>(x.size(dim));
    
    int64_t inner_size = 1;
    for(int i = dim + 1; i < dims; i++) inner_size *= x.size(i);
    
    std::vector<int64_t> out_sizes;
    for(int i = 0; i < dims; i++) {
        if(i != dim) out_sizes.push_back(x.size(i));
    }
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));
    
    cudaMemcpyToSymbol(const_mem_k, &K, sizeof(int));
    
    int threads = 256;
    dim3 block_dim(threads);
    dim3 grid_dim((inner_size + threads - 1) / threads, outer_size);
    
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        size_t shared_mem_size = threads * (sizeof(scalar_t) + sizeof(int));
        optimized_argmin_kernel<scalar_t><<<grid_dim, block_dim, shared_mem_size>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<int64_t>(),
            outer_size,
            inner_size
        );
    }));
    
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
    }
    
    return output;
}