#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// Warp size is fixed at 32 for current CUDA architectures
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int wid = threadIdx.x / WARP_SIZE;  // warp ID within block
    const int lane = threadIdx.x % WARP_SIZE; // lane ID within warp
    
    // Shared memory for partial sums - one per warp
    __shared__ scalar_t warp_sums[BLOCK_SIZE/WARP_SIZE];
    
    // Process elements that belong to the same output position
    if (tid < outer_size * inner_size) {
        const int outer_idx = tid / inner_size;
        const int inner_idx = tid % inner_size;
        const int64_t input_offset = outer_idx * dim_size * inner_size + inner_idx;
        
        // Each thread accumulates its portion
        scalar_t thread_sum = 0;
        
        // Align loads to warp boundaries to minimize divergence
        const int aligned_start = lane * ((dim_size + WARP_SIZE - 1) / WARP_SIZE);
        const int aligned_end = min(aligned_start + (dim_size + WARP_SIZE - 1) / WARP_SIZE, dim_size);
        
        // Main reduction loop - each thread processes its aligned portion
        #pragma unroll 4
        for (int i = aligned_start; i < aligned_end; i++) {
            thread_sum += __ldg(input + input_offset + i * inner_size);
        }
        
        // Warp-level reduction using shuffle operations
        thread_sum = warp_reduce_sum(thread_sum);
        
        // First thread in each warp writes the warp's sum
        if (lane == 0) {
            warp_sums[wid] = thread_sum;
        }
        __syncthreads();
        
        // First warp reduces the warp sums
        if (wid == 0) {
            scalar_t sum = 0;
            if (lane < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
                sum = warp_sums[lane];
            }
            
            sum = warp_reduce_sum(sum);
            
            if (lane == 0) {
                output[tid] = sum / static_cast<scalar_t>(dim_size);
            }
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
        mean_reduce_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &mean_reduce_cuda, "Mean reduction (CUDA)");
}