#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

template <typename scalar_t>
__global__ void argmin_parallel_reduction_kernel(const scalar_t* __restrict__ x,
                                           int64_t* __restrict__ output,
                                           int K,
                                           int64_t outer_size,
                                           int64_t inner_size) {
    extern __shared__ __align__(sizeof(int)) char shared_mem[];
    scalar_t* s_val = reinterpret_cast<scalar_t*>(shared_mem);
    int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

    const int64_t total_slices = outer_size * inner_size;
    const int64_t slice_idx = blockIdx.x;
    if (slice_idx >= total_slices) return;

    const int64_t outer = slice_idx / inner_size;
    const int64_t inner = slice_idx % inner_size;
    const scalar_t* slice_start = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;

    const int tid = threadIdx.x;
    scalar_t thread_min = slice_start[tid * inner_size];
    int thread_idx = tid;

    for (int k = tid + blockDim.x; k < K; k += blockDim.x) {
        scalar_t val = slice_start[k * inner_size];
        if (val < thread_min) {
            thread_min = val;
            thread_idx = k;
        }
    }

    s_val[tid] = thread_min;
    s_idx[tid] = thread_idx;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] < s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[slice_idx] = s_idx[0];
    }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

    int dims = x.dim();
    dim = (dim < 0) ? dim + dims : dim;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= x.size(i);
    const int K = static_cast<int>(x.size(dim));
    int64_t inner_size = 1;
    for (int i = dim + 1; i < dims; i++) inner_size *= x.size(i);

    std::vector<int64_t> out_sizes;
    for (int i = 0; i < dims; i++) if (i != dim) out_sizes.push_back(x.size(i));
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

    const int64_t total_slices = outer_size * inner_size;
    const int threads = 512;
    const size_t shared_mem = threads * (sizeof(scalar_t) + sizeof(int));

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half,
        x.scalar_type(),
        "argmin_cuda_forward",
        ([&] {
            argmin_parallel_reduction_kernel<scalar_t>
                <<<total_slices, threads, shared_mem>>>(
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
    m.def("forward", &argmin_cuda_forward, "Argmin forward (CUDA parallel reduction)");
}
