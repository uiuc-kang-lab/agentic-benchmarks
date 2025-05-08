#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Shared memory-based sum reduction device function
template <typename scalar_t>
__device__ inline scalar_t compute_sum(
    const scalar_t* input,
    int64_t base_idx,
    int64_t reduce_size,
    int64_t inner_size,
    scalar_t* shared_mem) {
    
    const int tid = threadIdx.x;
    scalar_t sum = 0;
    
    // First pass: thread-local reduction with unrolled loops
    #pragma unroll 4
    for (int64_t i = 0; i < reduce_size; i++) {
        sum += input[base_idx + i * inner_size];
    }
    
    // Store in shared memory
    shared_mem[tid] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    return shared_mem[0];
}

template <typename scalar_t>
__global__ void optimized_sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t outer_size,
    int64_t inner_size) {
    
    extern __shared__ char shared_mem_[];
    scalar_t* shared_mem = reinterpret_cast<scalar_t*>(shared_mem_);
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_size * inner_size) return;
    
    const int outer_idx = idx / inner_size;
    const int inner_idx = idx % inner_size;
    const int64_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;
    
    // Use shared memory-based reduction
    scalar_t block_sum = compute_sum(input, base_idx, reduce_size, inner_size, shared_mem);
    
    // Only the first thread in each block writes the result
    if (threadIdx.x == 0) {
        output[outer_idx * inner_size + inner_idx] = block_sum;
    }
}

torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];
    
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
    
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(typename std::iterator_traits<decltype(input.data_ptr<float>())>::value_type);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        optimized_sum_reduce_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            outer_size,
            inner_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Optimized sum reduction forward (CUDA)");
}