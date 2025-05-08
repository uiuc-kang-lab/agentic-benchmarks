#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel to compute reverse cumulative sum with memory access coalescing.
template <typename scalar_t>
__global__ void reverse_cumsum_kernel_coalesced(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t numel,
    int stride_dim,
    int n) {

    // Initialize shared memory for storing cumulative results
    __shared__ scalar_t sdata[256];  // Hard-coded to 256 threads
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int j = idx; j < numel; j += stride) {
        // Read input in reverse order and accumulate
        sdata[threadIdx.x] = (j / stride_dim == blockIdx.y) ? input[j] : 0;
        __syncthreads();

        // Reverse cumulative sum within the block
        scalar_t cum = 0;
        for (int i = (n - 1) - threadIdx.x; i >= 0; i -= blockDim.x) {
            cum += sdata[i];
        }

        // Write results to output
        output[j] = (__syncthreads(), cum);
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
    int64_t n = x.size(dim);
    int64_t numel = x.numel();
    int stride_dim = x.stride(dim);

    // Configure CUDA kernel launch parameters
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum_cuda_coalesced", ([&] {
        reverse_cumsum_kernel_coalesced<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel,
            stride_dim,
            n);
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with coalesced memory access (CUDA)");
}