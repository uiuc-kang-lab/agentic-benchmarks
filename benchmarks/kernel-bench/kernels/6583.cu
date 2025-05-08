#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined kernel using optimal block size and aligned global memory loads
// This kernel uses a templated block size and __ldg() for efficient memory access

template <typename scalar_t, int BLOCK_SIZE>
__global__ void optimized_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t num_outputs
) {
    // Process output elements in a grid-stride loop
    for (int out_idx = blockIdx.x; out_idx < num_outputs; out_idx += gridDim.x) {
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;
        int64_t base = outer_idx * dim_size * inner_size + inner_idx;

        int tid = threadIdx.x;
        bool valid = false;
        scalar_t thread_max = 0; // Will be overwritten by the first valid value
        // Each thread processes multiple elements in the reduction dimension in strides of BLOCK_SIZE
        for (int j = tid; j < dim_size; j += BLOCK_SIZE) {
            scalar_t val = __ldg(&input[base + j * inner_size]);
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
        shmem[tid] = valid ? thread_max : __ldg(&input[base]);
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

    int grid = (num_outputs < 1024) ? num_outputs : 1024;
    size_t shm_size = block_size * input.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "optimized_max_reduce_forward", ([&] {
        switch (block_size) {
            case 32:
                optimized_max_reduce_kernel<scalar_t, 32><<<grid, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
            case 64:
                optimized_max_reduce_kernel<scalar_t, 64><<<grid, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
            case 128:
                optimized_max_reduce_kernel<scalar_t, 128><<<grid, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
            case 256:
                optimized_max_reduce_kernel<scalar_t, 256><<<grid, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
            case 512:
                optimized_max_reduce_kernel<scalar_t, 512><<<grid, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
            default:
                optimized_max_reduce_kernel<scalar_t, 256><<<grid, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward, "Optimized max reduce forward (CUDA)");
}
