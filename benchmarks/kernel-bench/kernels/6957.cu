#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

template <typename scalar_t>
__global__ void argmin_warp_reduce_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    constexpr int warp_size = 32;
    
    // Calculate slice index
    int64_t slice_idx = bid;
    if (slice_idx >= outer_size * inner_size) return;
    
    // Precompute base pointer for this slice
    int64_t outer = slice_idx / inner_size;
    int64_t inner = slice_idx % inner_size;
    const scalar_t* slice_start = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;
    
    // Thread-local min tracking
    scalar_t min_val = INFINITY;
    int min_idx = 0;
    
    // Process elements with stride equal to blockDim.x
    for (int k = tid; k < K; k += blockDim.x) {
        scalar_t val = __ldg(slice_start + k * inner_size);
        if (val < min_val) {
            min_val = val;
            min_idx = k;
        }
    }
    
    // Warp-level reduction
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
        scalar_t other_val = __shfl_down_sync(0xffffffff, min_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, min_idx, offset);
        if (other_val < min_val || (other_val == min_val && other_idx < min_idx)) {
            min_val = other_val;
            min_idx = other_idx;
        }
    }
    
    // Final block reduction using shared memory
    __shared__ int s_min_indices[32];
    __shared__ scalar_t s_min_vals[32];
    
    if (tid % warp_size == 0) {
        s_min_vals[tid/warp_size] = min_val;
        s_min_indices[tid/warp_size] = min_idx;
    }
    __syncthreads();
    
    // First warp reduces partial results
    if (tid < warp_size) {
        scalar_t val = (tid < blockDim.x/warp_size) ? s_min_vals[tid] : INFINITY;
        int idx = (tid < blockDim.x/warp_size) ? s_min_indices[tid] : 0;
        
        for (int offset = warp_size/2; offset > 0; offset >>= 1) {
            scalar_t other_val = __shfl_down_sync(0xffffffff, val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
            if (other_val < val || (other_val == val && other_idx < idx)) {
                val = other_val;
                idx = other_idx;
            }
        }
        
        if (tid == 0) {
            output[slice_idx] = idx;
        }
    }
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
    
    // Tuned block size based on reduction dimension size
    int threads = K < 1024 ? 256 : 512;
    threads = K < 512 ? 128 : threads;
    threads = K < 256 ? 64 : threads;
    int blocks = outer_size * inner_size;
    
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_warp_reduce_kernel<scalar_t><<<blocks, threads>>>(x_data, output_data, K, outer_size, inner_size);
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