#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm>

template <typename scalar_t>
__device__ __forceinline__ const scalar_t* slice_ptr(const scalar_t* data, int64_t outer_idx, int64_t inner_idx, int64_t inner_size, int K) {
    return data + outer_idx * K * inner_size + inner_idx;
}

template <typename scalar_t>
__device__ int find_min_index(const scalar_t* slice, int K, int64_t stride) {
    scalar_t min_val = slice[0];
    int min_idx = 0;
    for(int k = 1; k < K; ++k) {
        scalar_t val = slice[k * stride];
        if(val < min_val) {
            min_val = val;
            min_idx = k;
        }
    }
    return min_idx;
}

template <typename scalar_t>
__global__ void argmin_kernel(const scalar_t* __restrict__ input,
                              int64_t* __restrict__ output,
                              int K,
                              int64_t outer_size,
                              int64_t inner_size) {
    const int64_t total = outer_size * inner_size;
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;

    for(int64_t i = idx; i < total; i += stride) {
        int64_t outer = i / inner_size;
        int64_t inner = i % inner_size;
        const scalar_t* slice = slice_ptr(input, outer, inner, K, inner_size);
        output[i] = find_min_index(slice, K, inner_size);
    }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be CUDA tensor");
    const int64_t ndim = x.dim();
    dim = dim < 0 ? dim + ndim : dim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dimension");

    int64_t outer = 1, inner = 1;
    for(int i = 0; i < dim; ++i) outer *= x.size(i);
    for(int i = dim + 1; i < ndim; ++i) inner *= x.size(i);
    const int K = x.size(dim);

    std::vector<int64_t> output_shape;
    for(int i = 0; i < ndim; ++i)
        if(i != dim) output_shape.push_back(x.size(i));
    auto output = at::empty(output_shape, x.options().dtype(at::kLong));

    const int64_t total = outer * inner;
    const int threads = 256;
    const int max_blocks = 2048;
    const int blocks = std::min<int>((total + threads - 1) / threads, max_blocks);

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_kernel", [&] {
        argmin_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<int64_t>(),
            K,
            outer,
            inner
        );
    });

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmin_cuda_forward, "Argmin (CUDA)");
}
