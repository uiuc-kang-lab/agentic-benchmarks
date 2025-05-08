#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel with minimal synchronization
// This kernel uses shared memory for block-level reduction with minimal __syncthreads() calls

template <typename scalar_t>
__global__ void optimized_shared_mem_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t num_outputs
) {
    extern __shared__ scalar_t shmax[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    for (int out_idx = blockIdx.x; out_idx < num_outputs; out_idx += gridDim.x) {
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;
        int64_t base = outer_idx * dim_size * inner_size + inner_idx;

        // Initialize shared memory with the first element
        scalar_t thread_max = input[base + tid * inner_size];

        // Each thread computes a partial maximum over its assigned segment
        for (int j = tid; j < dim_size; j += block_size) {
            scalar_t val = input[base + j * inner_size];
            thread_max = max(thread_max, val);
        }

        // Store the partial maximum in shared memory
        shmax[tid] = thread_max;
        __syncthreads();

        // Perform tree-based reduction in shared memory
        for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shmax[tid] = max(shmax[tid], shmax[tid + s]);
            }
            __syncthreads();
        }

        // The first thread in the block writes the result to the output
        if (tid == 0) {
            output[out_idx] = shmax[0];
        }
    }
}

// CUDA forward function
torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= input.size(i);
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) inner_size *= input.size(i);

    const int64_t dim_size = input.size(dim);
    const int64_t num_outputs = outer_size * inner_size;
    
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());
    
    int threads = (dim_size < 256 ? dim_size : 256);
    int blocks = (num_outputs < 1024 ? num_outputs : 1024);
    size_t shm_size = threads * sizeof(scalar_t);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "optimized_shared_mem_max_reduce_forward", ([&] {
        optimized_shared_mem_max_reduce_kernel<scalar_t><<<blocks, threads, shm_size>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Optimized shared memory max reduce forward (CUDA)");
}