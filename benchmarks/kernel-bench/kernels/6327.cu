#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Shared memory reduction kernel
template <typename scalar_t, int BLOCK_SIZE>
__global__ void sum_reduce_kernel_optimized(
    const scalar_t* input,
    scalar_t* output,
    int64_t reduce_size,
    int64_t outer_size,
    int64_t inner_size,
    int64_t offset,
    int64_t chunk_size) {
    
    __shared__ scalar_t shared_mem[BLOCK_SIZE];
    
    int global_idx = offset + blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= offset + chunk_size) return;

    int outer_idx = global_idx / inner_size;
    int inner_idx = global_idx % inner_size;
    
    // Load first element and accumulate remaining elements
    scalar_t sum = 0;
    int64_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;
    
    // Coalesced memory access with loop unrolling
    #pragma unroll 4
    for (int i = 0; i < reduce_size; i++) {
        sum += input[base_idx + i * inner_size];
    }
    
    shared_mem[threadIdx.x] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && global_idx + stride < offset + chunk_size) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Write result
    if (threadIdx.x == 0) {
        output[blockIdx.x] = shared_mem[0];
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
    
    int64_t total_elements = outer_size * inner_size;
    
    const int num_streams = (total_elements > 1048576) ? 4 : 1;
    int64_t chunk_size = (total_elements + num_streams - 1) / num_streams;
    
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    const int BLOCK_SIZE = 256;
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        for (int s = 0; s < num_streams; s++) {
            int64_t offset = s * chunk_size;
            int64_t current_chunk = std::min(chunk_size, total_elements - offset);
            if (current_chunk <= 0) break;
            
            int blocks = (current_chunk + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sum_reduce_kernel_optimized<scalar_t, BLOCK_SIZE>
                <<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(scalar_t), streams[s]>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                reduce_size,
                outer_size,
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