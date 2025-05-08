#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void reverse_cumsum_kernel(scalar_t* output, const scalar_t* input, 
                                    int size, int stride, int dim_size) {
    extern __shared__ scalar_t shared[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = bid * blockDim.x + tid;
    const int grid_stride = blockDim.x * gridDim.x;
    
    for (int base = gid; base < size; base += grid_stride) {
        // Calculate position in the dimension we're operating on
        int dim_pos = (base / stride) % dim_size;
        int offset = base - (dim_pos * stride);
        
        // Reverse the position
        int rev_pos = (dim_size - 1 - dim_pos) * stride + offset;
        
        // Load data into shared memory
        shared[tid] = input[rev_pos];
        __syncthreads();
        
        // Perform cumulative sum in shared memory
        scalar_t sum = shared[tid];
        for (int i = 1; i <= tid; i++) {
            sum += shared[tid - i];
        }
        
        // Write result to output
        output[base] = sum;
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    auto x_contig = x.contiguous();
    TORCH_CHECK(x_contig.is_cuda(), "Input tensor must be on CUDA");
    
    auto output = at::empty_like(x_contig);
    
    const int threads = 256;
    const int blocks = (x_contig.numel() + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(typename std::conditional<std::is_same<decltype(x_contig.scalar_type()), at::ScalarType::Float>::value, float, double>::type);
    
    AT_DISPATCH_FLOATING_TYPES(x_contig.scalar_type(), "reverse_cumsum_kernel", ([&] {
        reverse_cumsum_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            output.data_ptr<scalar_t>(),
            x_contig.data_ptr<scalar_t>(),
            x_contig.numel(),
            x_contig.stride(dim),
            x_contig.size(dim)
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum along a specified dimension (CUDA)");
}