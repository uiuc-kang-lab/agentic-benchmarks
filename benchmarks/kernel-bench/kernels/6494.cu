#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Template function to reduce with optimized workload distribution
template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {
    
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int block_threads = blockDim.x;
    const unsigned int output_idx = bid;
    
    if (output_idx >= outer_size * inner_size) return;
    
    const unsigned int outer_idx = output_idx / inner_size;
    const unsigned int inner_idx = output_idx % inner_size;
    const unsigned int input_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    // Load and sum elements with optimized workload distribution
    scalar_t thread_sum = 0;
    for (unsigned int i = tid; i < dim_size; i += block_threads) {
        if (!isnan(input[input_offset + i * inner_size])) {
        thread_sum += input[input_offset + i * inner_size];
    }
    }
    
    // Store the sum in shared memory
    shared_data[tid] = thread_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (unsigned int stride = block_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[output_idx] = shared_data[0] / static_cast<scalar_t>(dim_size);
    }
}

// Host function to prepare and launch the CUDA kernel
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
    
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    const int shared_mem_size = threads * input.element_size();
    
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
