#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Shared memory reduction kernel
template <typename scalar_t, unsigned int BLOCK_SIZE>
__global__ void sum_reduce_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t reduce_size,
    const int64_t inner_size,
    const int64_t offset,
    const int64_t chunk_size) {
    
    __shared__ scalar_t shared_mem[BLOCK_SIZE];
    
    const int tid = threadIdx.x;
    const int global_idx = offset + blockIdx.x * blockDim.x + tid;
    if (global_idx >= offset + chunk_size) return;

    const int outer_idx = global_idx / inner_size;
    const int inner_idx = global_idx % inner_size;
    const int64_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;
    
    // Load and compute initial sum
    scalar_t thread_sum = 0;
    #pragma unroll
    for (int i = 0; i < reduce_size; i++) {
        thread_sum += input[base_idx + i * inner_size];
    }
    
    shared_mem[tid] = thread_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    #pragma unroll
    for (unsigned int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s && global_idx + s < offset + chunk_size) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[global_idx] = shared_mem[0];
    }
}

torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    const int64_t reduce_size = sizes[dim];
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= sizes[i];
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) inner_size *= sizes[i];
    
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());
    
    const int64_t total_elements = outer_size * inner_size;
    const int num_streams = 4;  // Increased number of streams
    const int64_t chunk_size = (total_elements + num_streams - 1) / num_streams;
    
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    constexpr int BLOCK_SIZE = 256;
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        for (int s = 0; s < num_streams; s++) {
            const int64_t offset = s * chunk_size;
            const int64_t current_chunk = std::min(chunk_size, total_elements - offset);
            if (current_chunk <= 0) break;
            
            const int blocks = (current_chunk + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sum_reduce_kernel_optimized<scalar_t, BLOCK_SIZE>
                <<<blocks, BLOCK_SIZE, 0, streams[s]>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    reduce_size,
                    inner_size,
                    offset,
                    current_chunk
                );
        }
    }));
    
    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    
    return output;
}