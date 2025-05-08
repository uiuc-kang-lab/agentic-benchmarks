#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {
    
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int global_idx = bid * blockDim.x + tid;
    
    if (global_idx >= outer_size * inner_size) return;
    
    const int outer_idx = global_idx / inner_size;
    const int inner_idx = global_idx % inner_size;
    const int input_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    // First load and reduce into shared memory
    scalar_t thread_sum = 0;
    const int CHUNK_SIZE = (dim_size + blockDim.x - 1) / blockDim.x;
    
    #pragma unroll 4
    for (int chunk = 0; chunk < CHUNK_SIZE; chunk++) {
        const int dim_idx = chunk * blockDim.x + tid;
        if (dim_idx < dim_size) {
            thread_sum += __ldg(input + input_offset + dim_idx * inner_size);
        }
    }
    
    // Store partial sum in shared memory
    shared_data[tid] = thread_sum;
    __syncthreads();
    
    // Reduce within the block using shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < dim_size) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[global_idx] = shared_data[0] / static_cast<scalar_t>(dim_size);
    }
}

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];
    
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
    
    // Optimize thread block size for H100
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    // Calculate shared memory size
    const int shared_mem_size = threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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