#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void reverse_cumsum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t slice_size,
    const int64_t num_slices,
    const int64_t stride) {
    
    // Each block handles one complete slice
    const int64_t slice_idx = blockIdx.x;
    if (slice_idx >= num_slices) return;
    
    // Base offset for this slice
    const int64_t slice_offset = slice_idx * stride;
    
    // Shared memory for partial sums
    extern __shared__ char shared_mem[];
    scalar_t* partial_sums = reinterpret_cast<scalar_t*>(shared_mem);
    
    // Load data into shared memory with coalesced reads
    for (int idx = threadIdx.x; idx < slice_size; idx += blockDim.x) {
        partial_sums[idx] = input[slice_offset + idx];
    }
    __syncthreads();
    
    // Perform reverse cumsum in shared memory
    for (int idx = threadIdx.x; idx < slice_size; idx += blockDim.x) {
        scalar_t sum = 0;
        for (int j = slice_size - 1; j >= idx; --j) {
            sum += partial_sums[j];
        }
        output[slice_offset + idx] = sum;
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    
    // Handle dimension
    dim = dim < 0 ? dim + x.dim() : dim;
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "dim out of range");
    
    // Move dim to last position for coalesced access
    std::vector<int64_t> permute(x.dim());
    for (int64_t i = 0, j = 0; i < x.dim(); ++i) {
        if (i != dim) permute[j++] = i;
    }
    permute[x.dim() - 1] = dim;
    
    auto x_permuted = x.permute(permute).contiguous();
    auto output_permuted = at::empty_like(x_permuted);
    
    const int64_t slice_size = x.size(dim);
    const int64_t num_slices = x_permuted.numel() / slice_size;
    
    // Launch parameters
    const int threads_per_block = 256;
    const int blocks = num_slices;
    const int shared_mem_size = slice_size * sizeof(typename std::conditional<
        std::is_same<decltype(x_permuted.scalar_type()), at::ScalarType::Double>::value,
        double, float>::type);
    
    AT_DISPATCH_FLOATING_TYPES(x_permuted.scalar_type(), "reverse_cumsum_cuda", ([&] {
        reverse_cumsum_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_size>>>(
            x_permuted.data_ptr<scalar_t>(),
            output_permuted.data_ptr<scalar_t>(),
            slice_size,
            num_slices,
            slice_size  // stride is slice_size since tensor is contiguous
        );
    }));
    
    // Inverse permutation
    std::vector<int64_t> inv_permute(x.dim());
    for (int64_t i = 0; i < x.dim(); ++i) {
        inv_permute[permute[i]] = i;
    }
    
    return output_permuted.permute(inv_permute);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with coalesced memory access (CUDA)");
}