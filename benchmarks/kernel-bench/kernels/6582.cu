#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel: Each block computes the max reduction for one or more output elements using a tree-based reduction in shared memory.
// The block size is chosen adaptively at launch time based on the reduction dimension size.

template <typename scalar_t>
__global__ void optimal_block_size_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t num_outputs
) {
    // Each block processes output elements in a grid-stride manner.
    for (int out_idx = blockIdx.x; out_idx < num_outputs; out_idx += gridDim.x) {
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;
        int64_t base = outer_idx * dim_size * inner_size + inner_idx;

        int tid = threadIdx.x;
        int block_size = blockDim.x;

        // Each thread computes a partial maximum over a segment of the reduction dimension
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

        // Reduce partial results in shared memory
        extern __shared__ char sdata[];
        scalar_t* shmem = reinterpret_cast<scalar_t*>(sdata);

        // If no valid value was computed, default to the first element
        shmem[tid] = valid ? thread_max : input[base];
        __syncthreads();

        // Tree-based reduction in shared memory
        for (unsigned int stride = block_size / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                shmem[tid] = max(shmem[tid], shmem[tid + stride]);
            }
            __syncthreads();
        }

        // Write the reduction result to the output tensor
        if (tid == 0) {
            output[out_idx] = shmem[0];
        }
    }
}

// CUDA forward function
torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    // Calculate outer_size (product of dimensions before 'dim')
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    // Calculate inner_size (product of dimensions after 'dim')
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    const int64_t dim_size = input.size(dim);
    const int64_t num_outputs = outer_size * inner_size;

    // Prepare the output tensor by removing the reduced dimension
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());
    
    // Experiment with block sizes: choose among 32, 64, 128, 256, 512 based on dim_size
    int block_size = 32;
    if (dim_size >= 512) {
        block_size = 512;
    } else if (dim_size >= 256) {
        block_size = 256;
    } else if (dim_size >= 128) {
        block_size = 128;
    } else if (dim_size >= 64) {
        block_size = 64;
    } else {
        block_size = 32;
    }

    // If the number of outputs is smaller than the selected block size
    if (num_outputs < block_size) {
        block_size = num_outputs;
    }

    // Choose a reasonable number of blocks; cap at 1024 to balance kernel launch overhead
    int blocks = (num_outputs < 1024 ? num_outputs : 1024);

    // Allocate shared memory: one scalar_t per thread
    size_t shared_mem_size = block_size * input.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "optimal_block_size_max_reduce_forward", ([&] {
        optimal_block_size_max_reduce_kernel<scalar_t><<<blocks, block_size, shared_mem_size>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward (CUDA) with adaptive block size");
}
