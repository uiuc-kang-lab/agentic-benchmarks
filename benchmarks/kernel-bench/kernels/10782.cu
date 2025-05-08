#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t, int BLOCK_SIZE = 256>
__global__ void reverse_cumsum_kernel(const scalar_t* input, scalar_t* output, 
                                    const int64_t dim_size, const int64_t num_slices) {
    __shared__ scalar_t shared_data[BLOCK_SIZE];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    // Process one slice per block
    if (bid >= num_slices) return;
    
    const scalar_t* slice_in = input + bid * dim_size;
    scalar_t* slice_out = output + bid * dim_size;
    
    // Load slice into shared memory
    scalar_t running_sum = 0;
    for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
        shared_data[threadIdx.x] = slice_in[i];
        __syncthreads();
        
        // Compute running sum within shared memory
        if (i < dim_size) {
            running_sum += shared_data[threadIdx.x];
        }
        __syncthreads();
    }
    
    // First thread stores total sum
    if (tid == 0) {
        shared_data[0] = running_sum;
    }
    __syncthreads();
    
    // Compute reverse cumsum and write to output
    const scalar_t total = shared_data[0];
    for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
        if (i == 0) {
            slice_out[i] = total;
        } else {
            slice_out[i] = total - slice_in[i-1];
        }
    }
}

at::Tensor reverse_cumsum_optimized(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    
    const int64_t dim_size = x.size(dim);
    const int64_t num_slices = x.numel() / dim_size;
    
    auto output = torch::empty_like(x);
    
    const int threads = 256;
    const int blocks = num_slices;
    
    AT_DISPATCH_ALL_TYPES(x.scalar_type(), "reverse_cumsum", [&] {
        reverse_cumsum_kernel<scalar_t, 256><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            num_slices
        );
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum_optimized, "Optimized reverse cumsum with shared memory");
}