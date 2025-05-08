#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void optimized_parallel_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t num_outputs
) {
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_threads = blockDim.x;
    
    // Grid-stride loop over output elements
    for (int64_t out_idx = bid; out_idx < num_outputs; out_idx += gridDim.x) {
        const int64_t outer_idx = out_idx / inner_size;
        const int64_t inner_idx = out_idx % inner_size;
        const int64_t input_offset = outer_idx * dim_size * inner_size + inner_idx;
        
        // Initialize with first value each thread processes
        scalar_t thread_max;
        bool initialized = false;
        
        // Each thread processes multiple elements without synchronization
        #pragma unroll 4
        for (int i = tid; i < dim_size; i += num_threads) {
            scalar_t val = input[input_offset + i * inner_size];
            if (!initialized) {
                thread_max = val;
                initialized = true;
            } else {
                thread_max = max(thread_max, val);
            }
        }
        
        // Store thread's local maximum
        shared_data[tid] = initialized ? thread_max : input[input_offset];
        
        // Single sync point before reduction
        __syncthreads();
        
        // Tree-based reduction in shared memory with minimal syncs
        for (int stride = num_threads/2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
            }
            // Only synchronize if there are more iterations to come
            if (stride > 1) {
                __syncthreads();
            }
        }
        
        // Write result - no sync needed as we're done with shared memory
        if (tid == 0) {
            output[out_idx] = shared_data[0];
        }
    }
}

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }
    
    const int64_t dim_size = input.size(dim);
    const int64_t num_outputs = outer_size * inner_size;
    
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());
    
    // Optimize thread block size based on reduction size
    const int thread_block_size = 256;
    const int num_blocks = min(1024, static_cast<int>((num_outputs + thread_block_size - 1) / thread_block_size));
    const int shared_mem_size = thread_block_size * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        optimized_parallel_max_reduce_kernel<scalar_t><<<num_blocks, thread_block_size, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            inner_size,
            num_outputs
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward (CUDA)");
}