#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

template <typename scalar_t>
__global__ void argmin_warp_coalesced_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size) {
    
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int warp_id = tid >> 5;  // tid / 32
    const unsigned int lane_id = tid & 31;  // tid % 32
    const unsigned int warps_per_block = blockDim.x >> 5;
    
    // Shared memory for warp results
    __shared__ scalar_t s_min_vals[8];  // Assuming max 8 warps per block
    __shared__ int s_min_indices[8];
    
    // Calculate the base index for this thread block
    int64_t block_offset = bid * 32 * warps_per_block;
    int64_t thread_idx = block_offset + (warp_id * 32) + lane_id;
    
    if (thread_idx >= outer_size * inner_size) return;
    
    // Calculate outer and inner indices
    int64_t outer = thread_idx / inner_size;
    int64_t inner = thread_idx % inner_size;
    
    // Initialize with first value
    scalar_t local_min = x[outer * (K * inner_size) + inner];
    int local_min_idx = 0;
    
    // Process K dimension with coalesced accesses within each warp
    for (int k = 1; k < K; k++) {
        scalar_t val = x[outer * (K * inner_size) + k * inner_size + inner];
        if (val < local_min) {
            local_min = val;
            local_min_idx = k;
        }
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        scalar_t other_val = __shfl_down_sync(0xffffffff, local_min, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_min_idx, offset);
        
        if (lane_id < offset && other_val < local_min) {
            local_min = other_val;
            local_min_idx = other_idx;
        }
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        s_min_vals[warp_id] = local_min;
        s_min_indices[warp_id] = local_min_idx;
    }
    __syncthreads();
    
    // Final reduction across warps (done by first warp)
    if (warp_id == 0 && lane_id < warps_per_block) {
        local_min = s_min_vals[lane_id];
        local_min_idx = s_min_indices[lane_id];
        
        // Warp-level reduction for final results
        #pragma unroll
        for (int offset = warps_per_block >> 1; offset > 0; offset >>= 1) {
            scalar_t other_val = __shfl_down_sync(0xffffffff, local_min, offset);
            int other_idx = __shfl_down_sync(0xffffffff, local_min_idx, offset);
            
            if (lane_id < offset && other_val < local_min) {
                local_min = other_val;
                local_min_idx = other_idx;
            }
        }
        
        // Write final result
        if (lane_id == 0) {
            output[bid] = local_min_idx;
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
    
    const int threads_per_block = 256;  // Must be multiple of 32
    const int warps_per_block = threads_per_block / 32;
    const int num_blocks = (outer_size * inner_size + (threads_per_block - 1)) / threads_per_block;
    
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_warp_coalesced_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
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