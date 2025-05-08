#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void reverse_cumsum_kernel(float* x, float* out, int64_t size, int64_t dim_stride) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < size) {
        // Load data into shared memory
        shared_data[tid] = x[idx];
        __syncthreads();

        // Perform reverse cumulative sum in shared memory
        for (int offset = 1; offset < blockDim.x; offset *= 2) {
            float temp = 0.0f;
            if (tid >= offset) {
                temp = shared_data[tid - offset];
            }
            __syncthreads();
            shared_data[tid] += temp;
            __syncthreads();
        }

        // Write the result back to global memory
        out[idx] = shared_data[tid];
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