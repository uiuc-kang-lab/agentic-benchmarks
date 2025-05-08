#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void warp_sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t total_output) {

    // Each block handles one output element
    const int idx = blockIdx.x;
    const int lane = threadIdx.x;
    
    // Predicate for valid output indices - computed once per thread
    const bool valid_idx = idx < total_output;
    
    // Pre-compute indices only if the thread is working on valid data
    const int outer_idx = valid_idx ? (idx / inner_size) : 0;
    const int inner_idx = valid_idx ? (idx % inner_size) : 0;
    const int64_t base_offset = outer_idx * reduce_size * inner_size + inner_idx;
    
    // Initialize sum in register
    scalar_t sum = 0;
    
    // Compute number of full warps needed for reduction
    const int num_warps = (reduce_size + warpSize - 1) / warpSize;
    
    // Main reduction loop - all threads participate uniformly
    #pragma unroll 4
    for (int w = 0; w < num_warps; w++) {
        const int i = w * warpSize + lane;
        // Use predicated execution instead of branching
        const bool valid_load = i < reduce_size && valid_idx;
        const int64_t offset = base_offset + i * inner_size;
        // Predicated load - will be 0 for invalid indices
        const scalar_t val = valid_load ? input[offset] : 0;
        sum += val;
    }

    // Warp-level reduction using shuffle - all threads participate uniformly
    scalar_t partial = sum;
    const unsigned int mask = 0xffffffff;
    
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        const scalar_t n = __shfl_down_sync(mask, partial, offset);
        partial += n;
    }

    // Write result using predicated execution instead of branching
    // Only lane 0 will have valid data to write
    if (lane == 0 && valid_idx) {
        output[idx] = partial;
    }
}

torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    const int64_t reduce_size = sizes[dim];
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }
    
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());
    
    const int64_t total_output = outer_size * inner_size;
    const int threads = 64;  // increased thread count for better occupancy
    const int blocks = total_output;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        warp_sum_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size,
            total_output
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA)");
}