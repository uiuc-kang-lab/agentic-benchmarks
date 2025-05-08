#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use constant memory for frequently accessed read-only data
__constant__ int64_t const_dim_size;
__constant__ int64_t const_inner_size;

// Kernel using constant memory for dimension sizes
// Each block computes the max over the reduction dimension for one or more output elements

template <typename scalar_t, int BLOCK_SIZE>
__global__ void constant_memory_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t num_outputs
) {
    // Process output elements in a grid-stride loop
    for (int out_idx = blockIdx.x; out_idx < num_outputs; out_idx += gridDim.x) {
        int outer_idx = out_idx / const_inner_size;
        int inner_idx = out_idx % const_inner_size;
        // Compute the base index at the beginning of the reduction dimension
        int64_t base = outer_idx * const_dim_size * const_inner_size + inner_idx;

        int tid = threadIdx.x;
        bool valid = false;
        scalar_t thread_max = 0; // Will be overwritten by the first valid value
        // Each thread processes multiple elements in the reduction dimension in strides of BLOCK_SIZE
        for (int j = tid; j < const_dim_size; j += BLOCK_SIZE) {
            scalar_t val = input[base + j * const_inner_size];
            if (!valid) {
                thread_max = val;
                valid = true;
            } else {
                thread_max = max(thread_max, val);
            }
        }

        // Allocate shared memory for block-level reduction
        extern __shared__ char smem[];
        scalar_t* shmem = reinterpret_cast<scalar_t*>(smem);

        // For threads that did not process any element, use the first element as fallback
        shmem[tid] = valid ? thread_max : input[base];
        __syncthreads();

        // Tree-based reduction in shared memory
        for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shmem[tid] = max(shmem[tid], shmem[tid + s]);
            }
            __syncthreads();
        }

        // First thread writes the result
        if (tid == 0) {
            output[out_idx] = shmem[0];
        }
    }
}

// Forward function: determines optimal block size based on reduction dimension, then dispatches the templated kernel
// The heuristic selects among block sizes of 32, 64, 128, 256, and 512.

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    // Handle negative dimension
    if (dim < 0) dim += input.dim();

    // Compute outer_size (product of dimensions before 'dim') and inner_size (product after 'dim')
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

    // Copy dimension sizes to constant memory
    cudaMemcpyToSymbol(const_dim_size, &dim_size, sizeof(int64_t));
    cudaMemcpyToSymbol(const_inner_size, &inner_size, sizeof(int64_t));

    // Prepare output tensor with the reduced dimension removed
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // Heuristic to choose block size based on dim_size
    int block_size = 256;  // default value
    if (dim_size <= 32) {
        block_size = 32;
    } else if (dim_size <= 64) {
        block_size = 64;
    } else if (dim_size <= 128) {
        block_size = 128;
    } else if (dim_size <= 256) {
        block_size = 256;
    } else {
        block_size = 512;
    }

    // Choose grid size: limit grid.y to avoid launching too many blocks if not needed
    int grid = (num_outputs < 1024) ? num_outputs : 1024;

    // Shared memory size:
    size_t shm_size = block_size * input.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "constant_memory_max_reduce_forward", ([&] {
        switch (block_size) {
            case 32:
                constant_memory_max_reduce_kernel<scalar_t, 32><<<grid, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    num_outputs);
                break;
            case 64:
                constant_memory_max_reduce_kernel<scalar_t, 64><<<grid, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    num_outputs);
                break;
            case 128:
                constant_memory_max_reduce_kernel<scalar_t, 128><<<grid, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    num_outputs);
                break;
            case 256:
                constant_memory_max_reduce_kernel<scalar_t, 256><<<grid, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    num_outputs);
                break;
            case 512:
                constant_memory_max_reduce_kernel<scalar_t, 512><<<grid, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    num_outputs);
                break;
            default:
                // Fallback with default block size
                constant_memory_max_reduce_kernel<scalar_t, 256><<<grid, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    num_outputs);
                break;
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward with constant memory (CUDA)");
}
