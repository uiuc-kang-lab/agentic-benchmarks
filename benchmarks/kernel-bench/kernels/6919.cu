#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

template <typename scalar_t>
__global__ void argmin_kernel(const scalar_t* __restrict__ x,
                            int64_t* __restrict__ output,
                            int K,
                            int64_t outer_size,
                            int64_t inner_size) {
    extern __shared__ struct {
        scalar_t val;
        int idx;
    } shared[];

    int64_t tid = threadIdx.x;
    int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_slices = outer_size * inner_size;
    
    if (gid >= total_slices) return;

    // Decompose gid into outer and inner indices
    int64_t outer = gid / inner_size;
    int64_t inner = gid % inner_size;
    const scalar_t* slice_start = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;

    // Initialize with first element
    shared[tid].val = slice_start[0];
    shared[tid].idx = 0;

    // Find local minimum
    #pragma unroll
    for (int k = 1; k < K; ++k) {
        scalar_t val = slice_start[k * inner_size];
        if (val < shared[tid].val) {
            shared[tid].val = val;
            shared[tid].idx = k;
        }
    }

    // Write result directly if we're the only thread for this slice
    output[gid] = shared[tid].idx;
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

    int dims = x.dim();
    if (dim < 0) dim += dims;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }
    int K = static_cast<int>(x.size(dim));
    int64_t inner_size = 1;
    for (int i = dim + 1; i < dims; i++) {
        inner_size *= x.size(i);
    }

    std::vector<int64_t> out_sizes;
    for (int i = 0; i < dims; i++) {
        if (i == dim) continue;
        out_sizes.push_back(x.size(i));
    }
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

    int64_t total_slices = outer_size * inner_size;
    int threads = 256;
    int blocks = (total_slices + threads - 1) / threads;
    
    int shared_mem_size = threads * sizeof(struct {
        scalar_t val;
        int idx;
    });

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        argmin_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<int64_t>(),
            K,
            outer_size,
            inner_size
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmin_cuda_forward, "Argmin forward (CUDA)");
}