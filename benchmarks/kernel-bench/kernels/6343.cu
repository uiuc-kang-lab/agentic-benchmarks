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
    
    constexpr int BLOCK_WARPS = 8;  // Number of warps per block
    constexpr int WARP_SIZE = 32;
    constexpr int BLOCK_SIZE = BLOCK_WARPS * WARP_SIZE;
    constexpr int CHUNK_SIZE = 4;    // Number of elements per thread in initial load
    
    // Shared memory for block-level reduction
    extern __shared__ char smem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(smem);
    
    int outer_idx = blockIdx.x;
    int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    // Thread identifiers
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    scalar_t thread_sum = 0;
    
    if (inner_idx < inner_size) {
        // Each thread processes CHUNK_SIZE elements at a time
        int base_idx = outer_idx * (reduce_size * inner_size) + inner_idx;
        
        // Vectorized loading and initial reduction
        #pragma unroll
        for (int i = 0; i < reduce_size; i += CHUNK_SIZE * WARP_SIZE) {
            scalar_t chunk_vals[CHUNK_SIZE] = {0};
            
            // Load CHUNK_SIZE elements per thread
            #pragma unroll
            for (int j = 0; j < CHUNK_SIZE; j++) {
                int reduce_idx = i + lane_id + j * WARP_SIZE;
                if (reduce_idx < reduce_size) {
                    chunk_vals[j] = input[base_idx + reduce_idx * inner_size];
                }
            }
            
            // Sum the chunk
            #pragma unroll
            for (int j = 0; j < CHUNK_SIZE; j++) {
                thread_sum += chunk_vals[j];
            }
        }
        
        // Warp-level reduction using shuffle
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        
        // First thread in each warp writes to shared memory
        if (lane_id == 0) {
            shared_data[warp_id] = thread_sum;
        }
    }
    
    __syncthreads();
    
    // Final reduction across warps using the first warp
    if (warp_id == 0 && lane_id < BLOCK_WARPS) {
        thread_sum = shared_data[lane_id];
        
        // Warp-level reduction for final result
        for (int offset = BLOCK_WARPS/2; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        
        // Write final result
        if (lane_id == 0 && inner_idx < inner_size) {
            output[outer_idx * inner_size + inner_idx] = thread_sum;
        }
    }
}

torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];
    int64_t outer_size = 1;
    int64_t inner_size = 1;
    
    for (int i = 0; i < dim; i++) outer_size *= sizes[i];
    for (int i = dim + 1; i < sizes.size(); i++) inner_size *= sizes[i];
    
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());
    
    constexpr int BLOCK_SIZE = 256;
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(outer_size, (inner_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    size_t shared_mem_size = (BLOCK_SIZE / 32) * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        hybrid_reduce_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Hybrid sum reduction forward (CUDA)");
}