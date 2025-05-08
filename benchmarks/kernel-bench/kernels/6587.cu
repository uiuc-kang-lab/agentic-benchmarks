#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute the partial maximum for each thread over its assigned elements
template <typename scalar_t>
__device__ inline scalar_t compute_partial_max(const scalar_t* __restrict__ input,
                                                const int64_t base,
                                                const int dim_size,
                                                const int inner_size,
                                                const int block_size) {
    scalar_t thread_max;
    bool valid = false;
    int tid = threadIdx.x;
    // Each thread processes elements in strides of block_size
    for (int j = tid; j < dim_size; j += block_size) {
        scalar_t val = input[base + j * inner_size];
        if (!valid) {
            thread_max = val;
            valid = true;
        } else {
            thread_max = max(thread_max, val);
        }
    }
    return valid ? thread_max : input[base];
}

// Device function to perform tree-based reduction in shared memory
// Using volatile pointer to ensure proper memory accesses during reduction
template <typename scalar_t>
__device__ inline scalar_t reduce_shmem_max(volatile scalar_t* shmem, int block_size) {
    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride)
            shmem[threadIdx.x] = max(shmem[threadIdx.x], shmem[threadIdx.x + stride]);
        __syncthreads();
    }
    return shmem[0];
}

// Kernel that uses modular device functions for computing the max reduction
// Processes each output element via a grid-stride loop
template <typename scalar_t>
__global__ void modular_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t num_outputs
) {
    int block_size = blockDim.x;
    extern __shared__ char smem[];
    scalar_t* shmem = reinterpret_cast<scalar_t*>(smem);

    // Grid-stride loop over output elements
    for (int out_idx = blockIdx.x; out_idx < num_outputs; out_idx += gridDim.x) {
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;
        int64_t base = outer_idx * dim_size * inner_size + inner_idx;

        // Each thread computes its partial maximum over the reduction dimension
        scalar_t my_max = compute_partial_max<scalar_t>(input, base, dim_size, inner_size, block_size);

        // Store the partial result in shared memory
        shmem[threadIdx.x] = my_max;
        __syncthreads();

        // Perform a tree-based reduction to obtain the block maximum
        scalar_t block_max = reduce_shmem_max<scalar_t>(shmem, block_size);

        // Thread 0 writes the result for this output element
        if (threadIdx.x == 0) {
            output[out_idx] = block_max;
        }
        __syncthreads();
    }
}

// CUDA forward function that computes max reduction over the specified dimension
// Refactored to use modular device functions for improved readability and maintainability
torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0)
        dim += input.dim();

    // Compute outer_size (product of dimensions before 'dim')
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    // Compute inner_size (product of dimensions after 'dim')
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    const int64_t dim_size = input.size(dim);
    const int64_t num_outputs = outer_size * inner_size;

    // Prepare output tensor by removing the reduced dimension
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // Heuristic: choose block size based on the size of the reduction dimension
    int block_size = 256;  // default
    if (dim_size < 256) {
        if (dim_size <= 32)
            block_size = 32;
        else if (dim_size <= 64)
            block_size = 64;
        else if (dim_size <= 128)
            block_size = 128;
        else
            block_size = 256;
    }

    // Cap grid size to a reasonable number to balance launch overhead
    int grid = (num_outputs < 1024) ? num_outputs : 1024;
    size_t shared_mem_size = block_size * input.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "modular_max_reduce_forward", ([&] {
        modular_max_reduce_kernel<scalar_t><<<grid, block_size, shared_mem_size>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Modular max reduce forward (CUDA)");
}
