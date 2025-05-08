#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void reverse_cumsum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t numel,
    const int64_t slice_size) {
    
    extern __shared__ char shared_memory[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_memory);
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int slice_offset = bid * slice_size;
    
    // Load data into shared memory
    for (int i = tid; i < slice_size; i += block_size) {
        if (slice_offset + i < numel) {
            shared_data[i] = input[slice_offset + i];
        }
    }
    __syncthreads();
    
    // Compute reverse cumsum within shared memory
    for (int i = tid; i < slice_size; i += block_size) {
        if (slice_offset + i < numel) {
            scalar_t sum = 0;
            for (int j = i; j < slice_size; j++) {
                sum += shared_data[j];
            }
            output[slice_offset + i] = sum;
        }
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    
    // Handle last dimension as the most common case
    if (dim != x.dim() - 1) {
        x = x.transpose(dim, -1).contiguous();
    }
    
    auto output = at::empty_like(x);
    const int64_t slice_size = x.size(-1);
    const int64_t num_slices = x.numel() / slice_size;
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks = num_slices;
    const int shared_memory_size = slice_size * sizeof(typename std::conditional<
        std::is_same<decltype(x.scalar_type()), at::ScalarType::Float>::value,
        float, double>::type);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum_cuda", ([&] {
        reverse_cumsum_kernel<scalar_t><<<blocks, threads_per_block, shared_memory_size>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            x.numel(),
            slice_size
        );
    }));
    
    // Transpose back if necessary
    if (dim != x.dim() - 1) {
        output = output.transpose(dim, -1).contiguous();
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum using shared memory (CUDA)");
}