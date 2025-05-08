#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

template <typename scalar_t>
__global__ void argmin_warp_primitives_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid >> 5;  // tid / 32
    const int lane_id = tid & 31;  // tid % 32
    const int block_size = blockDim.x;
    
    // Calculate which slice this block is processing
    int64_t slice_idx = bid;
    if (slice_idx >= outer_size * inner_size) return;
    
    int64_t outer = slice_idx / inner_size;
    int64_t inner = slice_idx % inner_size;
    
    // Initialize with maximum value
    scalar_t local_min = FLT_MAX;
    int local_min_idx = 0;
    
    // Each thread processes elements strided by block_size
    for (int k = tid; k < K; k += block_size) {
        scalar_t val = __ldg(&x[outer * (K * inner_size) + k * inner_size + inner]);
        if (val < local_min) {
            local_min = val;
            local_min_idx = k;
        }
    }
    
    // Warp-level reduction using shuffle operations
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
    __shared__ scalar_t warp_mins[4];  // Assuming maximum 4 warps per block
    __shared__ int warp_indices[4];
    
    if (lane_id == 0) {
        warp_mins[warp_id] = local_min;
        warp_indices[warp_id] = local_min_idx;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp)
    if (warp_id == 0 && lane_id == 0) {
        scalar_t block_min = warp_mins[0];
        int block_min_idx = warp_indices[0];
        
        #pragma unroll
        for (int i = 1; i < (block_size + 31) / 32; i++) {
            if (warp_mins[i] < block_min) {
                block_min = warp_mins[i];
                block_min_idx = warp_indices[i];
            }
        }
        
        output[slice_idx] = block_min_idx;
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
    
    int threads = 128;  // Using 128 threads = 4 warps per block
    int blocks = outer_size * inner_size;
    
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_warp_primitives_kernel<scalar_t><<<blocks, threads>>>(
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