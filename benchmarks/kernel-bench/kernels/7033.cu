#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

template <typename scalar_t>
__global__ void argmin_kernel(const scalar_t* __restrict__ x,
                             int64_t* __restrict__ output,
                             int K,
                             int64_t inner_size,
                             int64_t outer_size) {
    const int inner = blockIdx.x * blockDim.x + threadIdx.x;
    const int outer = blockIdx.y;
    
    if (inner >= inner_size || outer >= outer_size) return;

    const scalar_t* slice_ptr = x + ((int64_t)outer * K * inner_size) + inner;
    scalar_t min_val = __ldg(slice_ptr);
    int min_idx = 0;

    for (int k = 1; k < K; ++k) {
        const scalar_t val = __ldg(slice_ptr + k * inner_size);
        const bool cond = val < min_val;
        min_idx = cond ? k : min_idx;
        min_val = cond ? val : min_val;
    }

    output[(int64_t)outer * inner_size + inner] = min_idx;
}

at::Tensor argmin_cuda_forward(const at::Tensor& x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    const int dims = x.dim();
    const int64_t dim_adj = dim < 0 ? dim + dims : dim;
    TORCH_CHECK(dim_adj >= 0 && dim_adj < dims, "Reduction dim out of range");

    int64_t outer_size = 1;
    for (int i = 0; i < dim_adj; ++i) outer_size *= x.size(i);
    const int K = x.size(dim_adj);
    int64_t inner_size = 1;
    for (int i = dim_adj+1; i < dims; ++i) inner_size *= x.size(i);

    std::vector<int64_t> out_shape;
    for (int i = 0; i < dims; ++i) if (i != dim_adj) out_shape.push_back(x.size(i));
    auto output = at::empty(out_shape, x.options().dtype(at::kLong));

    const int threads = 256;
    const dim3 blocks((inner_size + threads-1)/threads, outer_size);

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda", [&] {
        argmin_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<int64_t>(),
            K,
            inner_size,
            outer_size
        );
    });

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmin_cuda_forward, "Argmin forward (CUDA)");
}