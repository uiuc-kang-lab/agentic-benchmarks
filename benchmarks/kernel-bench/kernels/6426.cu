#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void cooperative_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ block_results,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t blocks_per_output) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int output_idx = bid / blocks_per_output;
    const int block_lane = bid % blocks_per_output;
    
    if (output_idx >= gridDim.x / blocks_per_output) return;
    
    const int outer_idx = output_idx / inner_size;
    const int inner_idx = output_idx % inner_size;
    const int64_t base = outer_idx * reduce_size * inner_size + inner_idx;
    
    // Each block handles a portion of the reduction dimension
    scalar_t sum = 0;
    
    // Grid-stride loop for better work distribution
    for (int i = block_lane * blockDim.x + tid; 
         i < reduce_size; 
         i += blockDim.x * blocks_per_output) {
        if (i < reduce_size) {
            sum += input[base + i * inner_size];
        }
    }
    
    // Warp reduction
    const unsigned int FULL_MASK = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    __shared__ scalar_t shared_data[32];
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    if (lane_id == 0) {
        shared_data[warp_id] = sum;
    }
    __syncthreads();
    
    // First warp reduces results from all warps
    if (warp_id == 0 && lane_id < (blockDim.x + 31) / 32) {
        sum = shared_data[lane_id];
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
        }
        
        if (lane_id == 0) {
            block_results[bid] = sum;
        }
    }
    
    // Final reduction of block results
    if (block_lane == 0 && tid == 0) {
        scalar_t final_sum = 0;
        for (int i = 0; i < blocks_per_output; i++) {
            final_sum += block_results[output_idx * blocks_per_output + i];
        }
        output[output_idx] = final_sum;
    }
}

torch::Tensor cooperative_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= sizes[i];
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) inner_size *= sizes[i];
    
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());
    
    const int64_t num_outputs = outer_size * inner_size;
    
    // Determine number of blocks per output based on reduction size
    const int blocks_per_output = std::min(32, static_cast<int>((reduce_size + 255) / 256));
    const int total_blocks = num_outputs * blocks_per_output;
    const int threads_per_block = 256;
    
    // Allocate temporary storage for block results
    auto block_results = torch::empty({total_blocks}, input.options());
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cooperative_reduce_cuda", ([&] {
        cooperative_reduce_kernel<scalar_t><<<total_blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            block_results.data_ptr<scalar_t>(),
            reduce_size,
            inner_size,
            blocks_per_output
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cooperative_reduce_cuda, "Cooperative block reduction (CUDA)");
}