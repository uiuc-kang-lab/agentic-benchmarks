#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void reverse_cumsum_kernel(float* x, float* out, int64_t size, int64_t dim_stride) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * blockDim.x + tid;
    
    // Calculate the starting position for this block in the dimension
    int dim_idx = gid / dim_stride;
    int inner_idx = gid % dim_stride;
    int dim_size = size / dim_stride;
    
    if (gid < size) {
        // Initialize output with input value
        out[gid] = x[gid];
        
        // Synchronize to ensure all threads have written their values
        __syncthreads();
        
        // Each thread accumulates values from previous positions in the same dimension
        for (int i = dim_idx + 1; i < dim_size; i++) {
            int prev_idx = i * dim_stride + inner_idx;
            if (prev_idx < size) {
                out[gid] += x[prev_idx];
            }
        }
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");

    auto x_flipped = x.flip(dim);
    auto out = at::empty_like(x);

    int64_t size = x_flipped.numel();
    int64_t dim_stride = x_flipped.stride(dim);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    reverse_cumsum_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        x_flipped.data_ptr<float>(), out.data_ptr<float>(), size, dim_stride);

    return out.flip(dim);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum along a specified dimension (CUDA)");
}