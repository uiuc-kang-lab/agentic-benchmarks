#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

template <typename scalar_t>
__global__ void argmin_shared_kernel(const scalar_t* __restrict__ x,
                                   int64_t* __restrict__ output,
                                   int K,
                                   int64_t outer_size,
                                   int64_t inner_size) {
    extern __shared__ char shared_mem[];
    scalar_t* shared_vals = reinterpret_cast<scalar_t*>(shared_mem);
    int* shared_indices = reinterpret_cast<int*>(shared_vals + blockDim.x);

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int64_t gid = static_cast<int64_t>(bid) * blockDim.x + tid;
    
    if (gid >= outer_size * inner_size) return;
    
    const int64_t outer = gid / inner_size;
    const int64_t inner = gid % inner_size;
    
    // Load first element and initialize
    const scalar_t* slice_ptr = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;
    scalar_t min_val = slice_ptr[0];
    int min_idx = 0;

    // First phase: each thread finds its local minimum
    for (int k = 1; k < K; k++) {
        scalar_t val = slice_ptr[k * inner_size];
        if (val < min_val) {
            min_val = val;
            min_idx = k;
        }
    }

    // Store local results in shared memory
    shared_vals[tid] = min_val;
    shared_indices[tid] = min_idx;
    __syncthreads();

    // Reduce within warps first
    const int warp_size = 32;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;
    const int warps_per_block = blockDim.x / warp_size;

    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        scalar_t other_val = __shfl_down_sync(0xffffffff, min_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, min_idx, offset);
        if (lane_id < offset && other_val < min_val) {
            min_val = other_val;
            min_idx = other_idx;
        }
    }

    // First thread in each warp writes its result
    if (lane_id == 0) {
        shared_vals[warp_id] = min_val;
        shared_indices[warp_id] = min_idx;
    }
    __syncthreads();

    // Final reduction across warps (done by first warp)
    if (tid < warps_per_block) {
        min_val = shared_vals[tid];
        min_idx = shared_indices[tid];
        
        #pragma unroll
        for (int offset = warps_per_block/2; offset > 0; offset /= 2) {
            scalar_t other_val = __shfl_down_sync(0xffffffff, min_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, min_idx, offset);
            if (lane_id < offset && other_val < min_val) {
                min_val = other_val;
                min_idx = other_idx;
            }
        }
    }

    // Write final result
    if (tid == 0) {
        output[gid] = min_idx;
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

    const int threads = 256;
    const int64_t total_elements = outer_size * inner_size;
    const int blocks = (total_elements + threads - 1) / threads;
    
    // Shared memory size per block
    const int shared_mem_size = threads * (sizeof(typename std::conditional<std::is_same<float, float>::value, float, double>::type) + sizeof(int));

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_shared_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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