#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses shared memory to coalesce reads for efficiency, 
// and synchronizes only when necessary.

template <typename scalar_t>
__global__ void sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* output,
    int64_t reduce_size,
    int64_t inner_size) {

    // Calculate the outer index each block is responsible for
    int outer_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Shared memory for storing partial sums
    extern __shared__ scalar_t ssum[];
    ssum[tid] = 0;

    int64_t base_offset = outer_idx * reduce_size * inner_size;

    // Unroll loop across the reduction dimension for coalesced access
    for (int i = 0; i < reduce_size; i++) {
        int64_t idx = base_offset + i * inner_size + tid;
        if (tid < inner_size) {
            ssum[tid] += input[idx];
        }
    }

    // Ensure all threads have finished writing to shared memory
    __syncthreads();

    // Write result to global memory
    if (tid < inner_size)
        output[outer_idx * inner_size + tid] = ssum[tid];
}

// CUDA wrapper
torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
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

    // Launch configuration: one block per outer index;
    // using threads per block to cover inner size with dynamic shared memory
    int threads = inner_size < 1024 ? inner_size : 1024;
    int blocks = outer_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        sum_reduce_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA)");
}
