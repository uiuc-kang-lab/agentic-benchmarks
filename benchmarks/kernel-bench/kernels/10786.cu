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
    
    // First compute the total sum of the slice
    scalar_t total_sum = 0;
    for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
        total_sum += slice_in[i];
    }
    
    // Use shared memory for block reduction to get total sum
    shared_data[tid] = total_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Get the total sum from shared memory
    const scalar_t total = shared_data[0];
    __syncthreads();
    
    // Compute reverse cumsum and write to output
    scalar_t running_sum = 0;
    for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
        // For each position, we want the sum of all elements from i onwards
        running_sum = 0;
        for (int j = i; j < dim_size; j++) {
            running_sum += slice_in[j];
        }
        slice_out[i] = running_sum;
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