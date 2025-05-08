#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Hybrid kernel combining shared memory tiling with warp-level primitives
template <typename scalar_t>
__global__ void hybrid_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size) {
    
    constexpr int WARP_SIZE = 32;
    constexpr int BLOCK_WARPS = 8;  // 8 warps per block = 256 threads
    constexpr int BLOCK_SIZE = WARP_SIZE * BLOCK_WARPS;
    
    // Shared memory for block-level reduction
    extern __shared__ char smem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(smem);
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int block_output_idx = blockIdx.x;
    
    // Calculate global indices
    int outer_idx = block_output_idx / inner_size;
    int inner_idx = block_output_idx % inner_size;
    
    // Each thread processes multiple elements with stride equal to block size
    scalar_t thread_sum = 0;
    int base_idx = outer_idx * (reduce_size * inner_size) + inner_idx;
    
    // Phase 1: Parallel reduction with loop unrolling
    #pragma unroll 4
    for (int i = tid; i < reduce_size; i += BLOCK_SIZE) {
        thread_sum += input[base_idx + i * inner_size];
    }
    
    // Phase 2: Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        shared_data[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Phase 3: Final reduction across warps using first warp
    if (warp_id == 0) {
        // Load warp results
        thread_sum = (lane_id < BLOCK_WARPS) ? shared_data[lane_id] : 0;
        
        // Final warp reduction
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        
        // Write final result
        if (lane_id == 0) {
            output[block_output_idx] = thread_sum;
        }
    }
}

torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= sizes[i];
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) inner_size *= sizes[i];
    
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());
    
    const int BLOCK_SIZE = 256;
    int total_outputs = outer_size * inner_size;
    int blocks = total_outputs;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        hybrid_reduce_kernel<scalar_t><<<blocks, BLOCK_SIZE, 
            (BLOCK_SIZE/32) * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Hybrid sum reduction (CUDA)");
}