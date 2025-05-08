#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel to compute reverse cumulative sum using shared memory for efficiency
// This kernel is optimized for the last dimension and uses shared memory to reduce global memory access latency.

template <typename scalar_t>
__global__ void reverse_cumsum_shared_kernel(const scalar_t* __restrict__ input,
                                             scalar_t* __restrict__ output,
                                             const int64_t n) {
    extern __shared__ scalar_t shared_data[];
    int row = blockIdx.x;
    const int64_t row_offset = row * n;
    const int tid = threadIdx.x;

    // Load data into shared memory in reverse order
    if (tid < n) {
        shared_data[n - 1 - tid] = input[row_offset + tid];
    }
    __syncthreads();

    // Perform reverse cumulative sum in shared memory
    scalar_t val = 0;
    if (tid < n) {
        for (int i = tid; i < n; ++i) {
            val += shared_data[i];
        }
        shared_data[tid] = val;
    }
    __syncthreads();

    // Write results back to global memory
    if (tid < n) {
        output[row_offset + tid] = shared_data[tid];
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    
    const int ndim = x.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    auto output = at::empty_like(x);

    int64_t n = x.size(dim);
    int64_t outer = x.numel() / n;

    if (dim == ndim - 1 && n <= 1024) {
        int threads = 1;
        while (threads < n) threads *= 2;
        if (threads > 1024) threads = 1024;

        dim3 blocks(outer);
        dim3 threadBlock(threads);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum_cuda_shared", ([&] {
            reverse_cumsum_shared_kernel<scalar_t><<<blocks, threadBlock, n * sizeof(scalar_t)>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                n);
        }));
    } else {
        auto x_flipped = x.flip(dim);
        auto cumsum = x_flipped.cumsum(dim);
        output = cumsum.flip(dim);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with shared memory optimization (CUDA)");
}