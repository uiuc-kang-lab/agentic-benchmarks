#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void reverseCumsumKernel(const float* __restrict__ x, float* __restrict__ out, int64_t size, int64_t dim_stride) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    // Load data into shared memory
    for (int i = idx; i < size; i += stride) {
        temp[tid] = x[i];
        __syncthreads();

        // Cumsum operation in reverse order
        for (int offset = 1; offset < blockDim.x; offset *= 2) {
            if (tid >= offset) {
                temp[tid] += temp[tid - offset];
            }
            __syncthreads();
        }

        out[i] = temp[tid];
    }
}

at::Tensor optimized_reverse_cumsum(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");

    int64_t size = x.size(dim);
    int64_t dim_stride = x.stride(dim);

    auto out = at::empty_like(x);

    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    // Calculate shared memory size
    size_t shared_mem_size = threads * sizeof(float);

    reverseCumsumKernel<<<blocks, threads, shared_mem_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size, dim_stride);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_reverse_cumsum, "Optimized Reverse cumulative sum along a specified dimension (CUDA)");
}