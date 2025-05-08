#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using shared memory for intra-block reduction and warp-level shuffles for the final reduction
template <typename scalar_t>
__global__ void max_reduce_shared_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t total_elements,
    const int64_t dim_size,
    const int64_t inner_size
) {
    extern __shared__ scalar_t shared_data[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_id = threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    scalar_t max_val = -FLT_MAX;

    for (int i = idx; i < total_elements; i += offset) {
        int outer_idx = i / inner_size;
        int inner_idx = i % inner_size;
        int64_t base = outer_idx * dim_size * inner_size + inner_idx;
        
        for (int j = 0; j < dim_size; j++) {
            scalar_t val = input[base + j * inner_size];
            max_val = max(max_val, val);
        }
    }

    shared_data[thread_id] = max_val;
    __syncthreads();

    // Intra-block reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_id < s) {
            shared_data[thread_id] = max(shared_data[thread_id], shared_data[thread_id + s]);
        }
        __syncthreads();
    }

    // Only one thread writes the result of the block-level reduction
    if (thread_id == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

__global__ void final_max_reduction(
    scalar_t* data,
    const int num_blocks
) {
    scalar_t max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        max_val = max(max_val, data[i]);
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    if (threadIdx.x == 0) {
        data[0] = max_val;
    }
}

// CUDA forward function
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

    int64_t total_elements = outer_size * inner_size;
    const int64_t dim_size = input.size(dim);

    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    const size_t shared_mem_size = threads * sizeof(scalar_t);

    auto intermediate_output = torch::empty({blocks}, input.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        max_reduce_shared_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            intermediate_output.data_ptr<scalar_t>(),
            total_elements,
            dim_size,
            inner_size
        );

        final_max_reduction<scalar_t><<<1, 32>>>(
            intermediate_output.data_ptr<scalar_t>(),
            blocks
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward (CUDA) with shared memory and warp-level optimization");
}
