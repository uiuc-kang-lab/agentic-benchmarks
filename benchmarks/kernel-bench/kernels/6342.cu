#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constants for optimization
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
constexpr int SHARED_MEM_ELEMENTS = BLOCK_SIZE;

template <typename scalar_t>
__device__ inline scalar_t warpReduceSum(scalar_t val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void hybrid_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t outer_size) {
    
    extern __shared__ char shared_memory[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_memory);
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int global_idx = blockIdx.x * BLOCK_SIZE + tid;
    
    // Calculate output position
    const int output_idx = global_idx / WARP_SIZE;
    if (output_idx >= outer_size * inner_size) return;
    
    const int outer_idx = output_idx / inner_size;
    const int inner_idx = output_idx % inner_size;
    
    // First phase: warp-level parallel reduction
    scalar_t thread_sum = 0;
    const int64_t base_idx = outer_idx * (reduce_size * inner_size) + inner_idx;
    
    // Each thread processes multiple elements with warp-level stride
    for (int i = lane_id; i < reduce_size; i += WARP_SIZE) {
        const int64_t input_idx = base_idx + i * inner_size;
        if (input_idx < outer_size * reduce_size * inner_size) {
            thread_sum += input[input_idx];
        }
    }
    
    // Perform warp-level reduction
    thread_sum = warpReduceSum(thread_sum);
    
    // Second phase: shared memory reduction across warps
    if (lane_id == 0) {
        shared_data[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction: first warp reduces results from all warps
    if (warp_id == 0) {
        scalar_t warp_sum = (lane_id < WARPS_PER_BLOCK) ? shared_data[lane_id] : 0;
        warp_sum = warpReduceSum(warp_sum);
        
        if (lane_id == 0 && output_idx < outer_size * inner_size) {
            output[output_idx] = warp_sum;
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
    
    const int64_t total_outputs = outer_size * inner_size;
    const int num_blocks = (total_outputs * WARP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hybrid_reduce_cuda", ([&] {
        hybrid_reduce_kernel<scalar_t><<<num_blocks, BLOCK_SIZE, SHARED_MEM_ELEMENTS * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size,
            outer_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Hybrid warp-shared memory reduction (CUDA)");
}