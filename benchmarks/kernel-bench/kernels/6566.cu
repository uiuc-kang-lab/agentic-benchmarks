#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel distributes the reduction workload of each output element across multiple threads
// using a block-level parallel reduction. A grid-stride loop over output elements ensures even
// distribution of work and avoids bottlenecks when the number of output elements is large.

template <typename scalar_t>
__global__ void max_reduce_parallel_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t num_outputs
) {
    // Each block processes one or more output elements using a grid-stride loop
    for (int out_idx = blockIdx.x; out_idx < num_outputs; out_idx += gridDim.x) {
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;
        // Compute the starting index for the reduction along the specified dimension
        int64_t base = outer_idx * dim_size * inner_size + inner_idx;

        int tid = threadIdx.x;
        int block_size = blockDim.x;
        
        // Each thread computes a partial maximum over its assigned segment of the reduction dimension
        bool valid = false;
        scalar_t thread_max;
        for (int j = tid; j < dim_size; j += block_size) {
            scalar_t val = input[base + j * inner_size];
            if (!valid) {
                thread_max = val;
                valid = true;
            } else {
                thread_max = max(thread_max, val);
            }
        }
        
        // Allocate shared memory for block-level reduction
        extern __shared__ char sdata[];
        scalar_t* shmax = reinterpret_cast<scalar_t*>(sdata);
        
        // Store the partial maximum; if a thread didn't process any element, use the first element
        shmax[tid] = valid ? thread_max : input[base];
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
    if (dim < 0)
        dim += input.dim();
    
    // Compute the product of dimensions before 'dim'
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }
    
    // Compute the product of dimensions after 'dim'
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    const int64_t dim_size = input.size(dim);
    
    // Total number of output elements after reducing the 'dim' dimension
    int64_t num_outputs = outer_size * inner_size;
    
    // Prepare output tensor by removing the reduced dimension
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());
    
    // Determine the number of threads per block based on the reduction size
    int threads = (dim_size < 256 ? dim_size : 256);
    // Use a moderate number of blocks to evenly cover all output elements
    int blocks = (num_outputs < 1024 ? num_outputs : 1024);
    
    // Allocate shared memory: one element per thread
    size_t shm_size = threads * input.element_size();
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_parallel_forward", ([&] {
        max_reduce_parallel_kernel<scalar_t><<<blocks, threads, shm_size>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward (CUDA) with distributed workload");
}
