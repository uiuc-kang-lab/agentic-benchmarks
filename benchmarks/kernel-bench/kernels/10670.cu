#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel to compute reverse cumulative sum along a given dimension.
// Supports arbitrary dimensions using grid-stride loops and correct boundary handling.

template <typename scalar_t>
__global__ void reverse_cumsum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer,    // number of slices (all dimensions except the one being cumsum-ed)
    int64_t n,        // size of the cumsum dimension
    int ndim,         // total number of dimensions of the tensor
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ strides,
    int dim) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t gridStride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple slices using a grid-stride loop
    for (int64_t r = idx; r < outer; r += gridStride) {
        int64_t offset = 0;
        // Fast-path for the common case when the cumsum dimension is the last dimension
        if (dim == ndim - 1) {
            // For a contiguous tensor with dim as the last dimension, slices are laid consecutively
            offset = r * n;
        } else {
            // For arbitrary dimension, compute the base offset for this slice by decomposing the
            // linear row index into multi-index (skipping the cumsum dimension).
            int64_t tmp = r;
            // Loop over dimensions in reverse order
            for (int d = ndim - 1; d >= 0; d--) {
                if (d == dim) continue;  // skip the cumsum dimension
                int64_t cur_size = sizes[d];
                int64_t idx_d = tmp % cur_size;
                tmp /= cur_size;
                offset += idx_d * strides[d];
            }
        }

        // Get the stride for the cumsum dimension. For contiguous tensors and dim==last, this is 1.
        int64_t stride_dim = (dim == ndim - 1) ? 1 : strides[dim];
        
        // Compute reverse cumulative sum for this slice
        scalar_t cum = scalar_t(0);
        for (int64_t j = n - 1; j >= 0; j--) {
            int64_t cur_index = offset + j * stride_dim;
            cum += input[cur_index];
            output[cur_index] = cum;
        }
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    // Ensure the tensor is contiguous and on CUDA
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    
    const int ndim = x.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    // Prepare output tensor
    auto output = at::empty_like(x);

    // Determine the size along the cumsum dimension and the number of slices (outer loops).
    int64_t n = x.size(dim);
    int64_t outer = x.numel() / n;

    // Configure CUDA kernel launch parameters
    const int threads = 256;
    const int blocks = (outer + threads - 1) / threads;

    // Prepare small arrays for tensor sizes and strides (maximum assumed dims = 10)
    const int max_dims = 10;
    int64_t h_sizes[max_dims];
    int64_t h_strides[max_dims];
    for (int i = 0; i < ndim; i++) {
        h_sizes[i] = x.size(i);
        h_strides[i] = x.stride(i);
    }

    // Allocate device memory for sizes and strides
    int64_t *d_sizes = nullptr, *d_strides = nullptr;
    cudaError_t err = cudaMalloc(&d_sizes, ndim * sizeof(int64_t));
    TORCH_CHECK(err == cudaSuccess, "cudaMalloc for d_sizes failed");
    err = cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    TORCH_CHECK(err == cudaSuccess, "cudaMalloc for d_strides failed");

    err = cudaMemcpy(d_sizes, h_sizes, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy for d_sizes failed");
    err = cudaMemcpy(d_strides, h_strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy for d_strides failed");

    // Dispatch kernel based on the scalar type
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum_cuda", ([&] {
        reverse_cumsum_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer,
            n,
            ndim,
            d_sizes,
            d_strides,
            dim);
    }));

    // Free temporary device memory
    cudaFree(d_sizes);
    cudaFree(d_strides);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with grid-stride loop (CUDA)");
}
