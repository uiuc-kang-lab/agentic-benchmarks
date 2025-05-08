#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

template <typename scalar_t>
__device__ __forceinline__ void initialize_local_min(scalar_t& local_min, int& local_min_idx) {
    local_min = CUDART_INF_F;
    local_min_idx = -1;
}

template <typename scalar_t>
__device__ __forceinline__ void process_elements(
    const scalar_t* slice_ptr,
    scalar_t& local_min,
    int& local_min_idx,
    int K,
    int tid,
    int block_size
) {
    for (int k = tid; k < K; k += block_size) {
        scalar_t val = __ldg(slice_ptr + k);
        if (val < local_min) {
            local_min = val;
            local_min_idx = k;
        }
    }
}

template <typename scalar_t>
__device__ __forceinline__ void block_reduce(
    scalar_t* s_min_vals,
    int* s_min_indices,
    scalar_t& local_min,
    int& local_min_idx,
    int tid
) {
    s_min_vals[tid] = local_min;
    s_min_indices[tid] = local_min_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            if (s_min_vals[tid + stride] < s_min_vals[tid]) {
                s_min_vals[tid] = s_min_vals[tid + stride];
                s_min_indices[tid] = s_min_indices[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid < 32) {
        for (int stride = 16; stride > 0; stride >>= 1) {
            scalar_t other_val = __shfl_down_sync(0xffffffff, s_min_vals[tid], stride);
            int other_idx = __shfl_down_sync(0xffffffff, s_min_indices[tid], stride);
            if (other_val < s_min_vals[tid]) {
                s_min_vals[tid] = other_val;
                s_min_indices[tid] = other_idx;
            }
        }
    }
}

template <typename scalar_t>
__global__ void argmin_modular_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size
) {
    const int tid = threadIdx.x;
    const int64_t slice_idx = blockIdx.x;
    if (slice_idx >= outer_size * inner_size) return;

    int64_t outer = slice_idx / inner_size;
    int64_t inner = slice_idx % inner_size;
    const scalar_t* slice_ptr = x + outer * (K * inner_size) + inner;

    scalar_t local_min;
    int local_min_idx;
    __shared__ scalar_t s_min_vals[128];
    __shared__ int s_min_indices[128];

    initialize_local_min(local_min, local_min_idx);
    process_elements(slice_ptr, local_min, local_min_idx, K, tid, blockDim.x);
    block_reduce(s_min_vals, s_min_indices, local_min, local_min_idx, tid);

    if (tid == 0) {
        output[slice_idx] = s_min_indices[0];
    }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

    int dims = x.dim();
    if (dim < 0) dim += dims;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= x.size(i);
    int K = x.size(dim);
    int64_t inner_size = 1;
    for (int i = dim + 1; i < dims; i++) inner_size *= x.size(i);

    std::vector<int64_t> out_sizes;
    for (int i = 0; i < dims; i++) if (i != dim) out_sizes.push_back(x.size(i));
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

    const int threads = 128;
    const int blocks = outer_size * inner_size;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        argmin_modular_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<int64_t>(),
            K,
            outer_size,
            inner_size
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmin_cuda_forward, "Argmin forward (CUDA)");
}
