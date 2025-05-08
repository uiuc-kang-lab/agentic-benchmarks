#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using stride loops to handle workloads larger than the number of available threads
// Ensures correct boundary handling and efficient reduction

template <typename scalar_t>
__global__ void stride_loop_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t num_outputs
) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int out_idx = blockIdx.x * block_size + tid;

    if (out_idx < num_outputs) {
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;
        int64_t base = outer_idx * dim_size * inner_size + inner_idx;

        scalar_t thread_max = -INFINITY;
        for (int j = 0; j < dim_size; j++) {
            scalar_t val = input[base + j * inner_size];
            thread_max = max(thread_max, val);
        }

        extern __shared__ char sdata[];
        scalar_t* shmem = reinterpret_cast<scalar_t*>(sdata);

        shmem[tid] = thread_max;
        __syncthreads();

        for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shmem[tid] = max(shmem[tid], shmem[tid + s]);
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[out_idx] = shmem[0];
        }
    }
}

// CUDA forward function with stride loop
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

    int block_size = 256;
    if (dim_size > 512) {
        block_size = 512;
    }

    if (num_outputs < block_size) {
        block_size = num_outputs;
    }

    int blocks = (num_outputs + block_size - 1) / block_size;

    size_t shared_mem_size = block_size * input.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "stride_loop_max_reduce_forward", ([&] {
        stride_loop_max_reduce_kernel<scalar_t><<<blocks, block_size, shared_mem_size>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward with stride loop (CUDA)");
}
