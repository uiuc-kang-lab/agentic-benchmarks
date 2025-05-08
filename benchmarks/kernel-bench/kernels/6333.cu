#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that leverages shared memory to cache input blocks and reduce memory latency during reduction.

template <typename scalar_t>
__global__ void sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t outer_size) {

    // Define shared memory to cache data used in reduction in each block
    extern __shared__ scalar_t shared_data[];

    int block = blockIdx.x;
    int tid = threadIdx.x;

    int outer_idx = block;
    int thread_count = blockDim.x;

    // Compute base offsets for this block
    int64_t base_input_offset = outer_idx * reduce_size * inner_size;
    int64_t base_output_offset = outer_idx * inner_size;

    // Each thread computes a subset of the output
    for (int inner_idx = tid; inner_idx < inner_size; inner_idx += thread_count) {
        scalar_t sum = 0;

        // Load data into shared memory in chunks and perform reduction on them
        for (int i = 0; i < reduce_size; i++) {
            if (tid < inner_size) {
                shared_data[tid] = input[base_input_offset + i * inner_size + inner_idx];
            }

            // Synchronize to make sure the data is loaded
            __syncthreads();

            // Sum the loaded data
            if (tid < inner_size) {
                sum += shared_data[tid];
            }

            // Synchronize again before next loop iteration to avoid data races
            __syncthreads();
        }

        if (tid < inner_size) {
            output[base_output_offset + inner_idx] = sum;
        }
    }
}

// CUDA wrapper function
torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
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

    // Set output size: the reduction dimension becomes 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Launch configuration
    const int threads = 256; // Choosing a reasonable block size
    const int blocks = outer_size;

    // Calculate shared memory size based on inner size
    int shared_mem_size = inner_size * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        sum_reduce_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size,
            outer_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA)");
}