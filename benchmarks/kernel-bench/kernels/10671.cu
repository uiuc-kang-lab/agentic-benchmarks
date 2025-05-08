#include <torch/extension.h>

__global__ void reverse_cumsum_kernel(float* x, float* out, int64_t size, int64_t stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = idx * stride;
    if (tid < size) {
        float sum = 0.0f;
        for (int i = tid; i < size; i += stride) {
            sum += x[i];
            out[i] = sum;
        }
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");

    auto x_flipped = x.flip(dim);
    auto out = at::empty_like(x);

    int64_t size = x_flipped.numel();
    int64_t threads = 256;
    int64_t blocks = (size + threads - 1) / threads;
    int64_t stride = gridDim.x * blockDim.x;

    reverse_cumsum_kernel<<<blocks, threads>>>(x_flipped.data_ptr<float>(), out.data_ptr<float>(), size, stride);

    cudaDeviceSynchronize();

    return out.flip(dim);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum along a specified dimension (CUDA)");
}