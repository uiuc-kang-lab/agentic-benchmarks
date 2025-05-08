#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

template <typename scalar_t>
__global__ void argmin_warp_optimized_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size) {
    
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int wid = tid / warpSize;
    const unsigned int lane = tid % warpSize;
    const unsigned int warps_per_block = blockDim.x / warpSize;
    
    // Shared memory for partial results
    __shared__ scalar_t s_min_vals[8][32]; // For 8 warps
    __shared__ int s_min_indices[8][32];
    
    int64_t slice_idx = bid;
    if (slice_idx >= outer_size * inner_size) return;
    
    int64_t outer = slice_idx / inner_size;
    int64_t inner = slice_idx % inner_size;
    
    // Initialize with the first value
    scalar_t local_min = FLT_MAX;
    int local_min_idx = 0;
    
    // Each thread processes multiple elements strided by blockDim.x
    for (int k = tid; k < K; k += blockDim.x) {
        scalar_t val = __ldg(&x[outer * (K * inner_size) + k * inner_size + inner]);
        if (val < local_min) {
            local_min = val;
            local_min_idx = k;
        }
    }
    
    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        scalar_t other_val = __shfl_down_sync(0xffffffff, local_min, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_min_idx, offset);
        if (lane < offset && other_val < local_min) {
            local_min = other_val;
            local_min_idx = other_idx;
        }
    }
    
    // Store warp results in shared memory
    if (lane == 0) {
        s_min_vals[wid][0] = local_min;
        s_min_indices[wid][0] = local_min_idx;
    }
    __syncthreads();
    
    // Final reduction across warps using the first warp
    if (wid == 0 && lane < warps_per_block) {
        scalar_t warp_min = s_min_vals[lane][0];
        int warp_min_idx = s_min_indices[lane][0];
        
        // Warp-level reduction of the final results
        #pragma unroll
        for (int offset = warps_per_block/2; offset > 0; offset /= 2) {
            scalar_t other_val = __shfl_down_sync(0xffffffff, warp_min, offset);
            int other_idx = __shfl_down_sync(0xffffffff, warp_min_idx, offset);
            if (lane < offset && other_val < warp_min) {
                warp_min = other_val;
                warp_min_idx = other_idx;
            }
        }
        
        // Write final result
        if (lane == 0) {
            output[slice_idx] = warp_min_idx;
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
    
    // Use 256 threads (8 warps) per block
    int threads = 256;
    int blocks = outer_size * inner_size;
    
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_warp_optimized_kernel<scalar_t><<<blocks, threads>>>(
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