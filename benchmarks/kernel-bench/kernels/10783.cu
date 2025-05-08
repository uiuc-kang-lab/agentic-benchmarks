#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>

// Kernel using shared memory: each block processes one slice (i.e., one contiguous segment along the target dimension).
// It loads the entire slice from global memory into shared memory, then computes the reverse cumulative sum
// (i.e., output[i] = total - (i > 0 ? shared[i-1] : 0)) with lower latency.

template <typename scalar_t>
__global__ void subtract_shifted_cumsum_shared(const scalar_t* __restrict__ cumsum,
                                               scalar_t* __restrict__ output,
                                               int64_t dim_size) {
    // Each block processes one slice.
    int64_t slice = blockIdx.x;
    const scalar_t* slice_cumsum = cumsum + slice * dim_size;
    scalar_t* slice_output = output + slice * dim_size;

    extern __shared__ scalar_t s_data[]; // dynamically-allocated shared memory for one slice

    // Load the entire slice into shared memory collaboratively
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        s_data[i] = slice_cumsum[i];
    }
    __syncthreads();

    // The total sum for this slice is at the last index in shared memory
    scalar_t total = s_data[dim_size - 1];

    // Compute the reverse cumulative sum using shared memory
    // For index 0, there is no preceding element; for others, subtract the previous cumsum.
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        if (i == 0) {
            slice_output[i] = total;
        } else {
            slice_output[i] = total - s_data[i - 1];
        }
    }
}

at::Tensor reverse_cumsum_optimized_shared(at::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");

    // If the target dimension is not the last dimension, permute so that it becomes contiguous
    bool permuted = false;
    std::vector<int64_t> orig_perm;
    if (dim != x.dim() - 1) {
        std::vector<int64_t> permute_dims(x.dim());
        std::iota(permute_dims.begin(), permute_dims.end(), 0);
        std::swap(permute_dims[dim], permute_dims[x.dim() - 1]);
        orig_perm = permute_dims;
        x = x.permute(permute_dims).contiguous();
        permuted = true;
        dim = x.dim() - 1;  // now the target dimension is the last one
    } else {
        x = x.contiguous();
    }

    // Compute cumulative sum along the (last) dimension
    auto cumsum = x.cumsum(dim);
    auto output = torch::empty_like(cumsum);

    int64_t dim_size = cumsum.size(dim);
    // Number of slices is the total number of elements divided by the size along the target dimension
    int64_t num_slices = cumsum.numel() / dim_size;

    // Launch parameters:
    // Each block processes one slice from cumsum, so grid.x = num_slices.
    // Use min(dim_size, 256) threads per block; if the slice is longer, threads loop over the slice.
    int threads = (dim_size < 256) ? static_cast<int>(dim_size) : 256;
    int blocks = num_slices;

    AT_DISPATCH_ALL_TYPES(x.scalar_type(), "reverse_cumsum_shared", ([&] {
        size_t shared_mem_bytes = sizeof(scalar_t) * dim_size;
        subtract_shifted_cumsum_shared<scalar_t><<<blocks, threads, shared_mem_bytes>>>(
            cumsum.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size
        );
    }));

    // If a permutation was applied to make the target dimension contiguous, revert it
    if (permuted) {
        std::vector<int64_t> inv_perm(orig_perm.size());
        for (size_t i = 0; i < orig_perm.size(); i++) {
            inv_perm[orig_perm[i]] = i;
        }
        output = output.permute(inv_perm).contiguous();
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum_optimized_shared, "Optimized reverse cumulative sum using shared memory");
}
