#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// Optimized CUDA kernel with balanced workload distribution

template <typename scalar_t>
__global__ void argmin_balanced_workload_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    
    // Shared memory for partial results
    __shared__ scalar_t s_min_vals[256];
    __shared__ int s_min_indices[256];
    
    // Calculate which slice this block is processing
    int64_t slice_idx = bid;
    if (slice_idx >= outer_size * inner_size) return;
    
    int64_t outer = slice_idx / inner_size;
    int64_t inner = slice_idx % inner_size;
    
    // Initialize with the first value each thread can access
    scalar_t local_min = FLT_MAX;
    int local_min_idx = 0;
    
    // Distribute workload evenly
    int elements_per_thread = (K + block_size - 1) / block_size;
    int start_k = tid * elements_per_thread;
    int end_k = min(start_k + elements_per_thread, K);
    
    // Each thread processes its assigned elements
    for (int k = start_k; k < end_k; ++k) {
        scalar_t val = __ldg(&x[outer * (K * inner_size) + k * inner_size + inner]);
        if (val < local_min) {
            local_min = val;
            local_min_idx = k;
        }
    }
    
    // Store in shared memory
    s_min_vals[tid] = local_min;
    s_min_indices[tid] = local_min_idx;
    __syncthreads();
    
    // Reduce within the block
    for (int stride = block_size/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_min_vals[tid + stride] < s_min_vals[tid]) {
                s_min_vals[tid] = s_min_vals[tid + stride];
                s_min_indices[tid] = s_min_indices[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // Write result
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
    
    int threads = 256;
    int blocks = outer_size * inner_size;
    
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_balanced_workload_kernel<scalar_t><<<blocks, threads>>>(
            x_data, output_data, K, outer_size, inner_size);
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
