#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <limits>

template <typename scalar_t>
__global__ void argmin_warp_kernel(const scalar_t* __restrict__ x,
                                 int64_t* __restrict__ output,
                                 int K,
                                 int64_t inner_size) {
    const int tid = threadIdx.x;
    const int wid = tid / warpSize;
    const int lane = tid % warpSize;
    const int warps_per_block = blockDim.x / warpSize;
    
    const int outer = blockIdx.y;
    const int inner = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (inner >= inner_size) return;
    
    const scalar_t* slice_start = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;
    
    // Each thread finds its local minimum
    scalar_t local_min = std::numeric_limits<scalar_t>::max();
    int local_idx = 0;
    
    #pragma unroll 4
    for (int k = 0; k < K; k++) {
        scalar_t val = slice_start[k * inner_size];
        if (val < local_min) {
            local_min = val;
            local_idx = k;
        }
    }
    
    // Warp reduction using shuffle
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        scalar_t other_val = __shfl_down_sync(0xffffffff, local_min, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        if (lane < offset && other_val < local_min) {
            local_min = other_val;
            local_idx = other_idx;
        }
    }
    
    // First thread in each warp writes to shared memory
    __shared__ scalar_t warp_mins[32];  // Max 32 warps per block
    __shared__ int warp_indices[32];
    
    if (lane == 0) {
        warp_mins[wid] = local_min;
        warp_indices[wid] = local_idx;
    }
    
    __syncthreads();
    
    // Final reduction (first warp only)
    if (wid == 0 && lane < warps_per_block) {
        local_min = warp_mins[lane];
        local_idx = warp_indices[lane];
        
        #pragma unroll
        for (int offset = (warps_per_block + 1)/2; offset > 0; offset /= 2) {
            scalar_t other_val = __shfl_down_sync(0xffffffff, local_min, offset);
            int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
            if (lane < offset && other_val < local_min) {
                local_min = other_val;
                local_idx = other_idx;
            }
        }
        
        if (lane == 0) {
            output[outer * inner_size + inner] = local_idx;
        }
    }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    int dims = x.dim();
    if (dim < 0) dim += dims;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= x.size(i);
    
    int K = static_cast<int>(x.size(dim));
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < dims; i++) inner_size *= x.size(i);
    
    std::vector<int64_t> out_sizes;
    for (int i = 0; i < dims; i++) {
        if (i == dim) continue;
        out_sizes.push_back(x.size(i));
    }
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));
    
    const int threads_per_block = 256;
    dim3 block_dim(threads_per_block);
    dim3 grid_dim((inner_size + threads_per_block - 1) / threads_per_block, outer_size);
    
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_warp_kernel<scalar_t><<<grid_dim, block_dim>>>(x_data, output_data, K, inner_size);
    }));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmin_cuda_forward, "Argmin forward with warp optimizations (CUDA)");
}