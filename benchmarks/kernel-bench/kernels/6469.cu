#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void mean_reduce_kernel_warp(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    
    __shared__ scalar_t warp_sums[WARP_SIZE];  // Space for warp partial sums
    
    if (tid >= outer_size * inner_size) return;
    
    const int outer_idx = tid / inner_size;
    const int inner_idx = tid % inner_size;
    const int base_offset = outer_idx * (dim_size * inner_size) + inner_idx;
    
    // Each thread accumulates its portion of the reduction dimension
    scalar_t thread_sum = 0;
    
    // Aligned reads for consecutive threads within a warp
    #pragma unroll 4
    for (int i = lane_id; i < dim_size; i += WARP_SIZE) {
        thread_sum += __ldg(input + base_offset + i * inner_size);
    }
    
    // Warp-level reduction using shuffle operations
    thread_sum = warp_reduce_sum(thread_sum);
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // First warp reduces results from all warps
    if (warp_id == 0) {
        thread_sum = (lane_id < warps_per_block) ? warp_sums[lane_id] : 0;
        thread_sum = warp_reduce_sum(thread_sum);
        
        if (lane_id == 0) {
            output[tid / WARP_SIZE] = thread_sum / static_cast<scalar_t>(dim_size);
        }
    }
}

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    const int64_t dim_size = sizes[dim];
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }
    
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());
    
    const int threads = BLOCK_SIZE;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel_warp<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Warp-optimized mean reduction (CUDA)");
}