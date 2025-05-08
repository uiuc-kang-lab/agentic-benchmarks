#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined kernel using manual unrolling and dynamic shared memory for partial reduction
template <typename scalar_t>
__global__ void max_reduce_combined_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    extern __shared__ scalar_t sdata[];  // Dynamic shared memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int total_elements = outer_size * inner_size;
    if (idx >= total_elements) return;

    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    // Compute starting index for this reduction
    int64_t base = outer_idx * dim_size * inner_size + inner_idx;

    // Initialize max_val with the first element
    scalar_t max_val = input[base];

    // Unroll reduction loop in steps of 4
    for (int i = 1; i + 3 < dim_size; i += 4) {
        scalar_t a = input[base + i * inner_size];
        scalar_t b = input[base + (i+1) * inner_size];
        scalar_t c = input[base + (i+2) * inner_size];
        scalar_t d = input[base + (i+3) * inner_size];
        max_val = max(max_val, a);
        max_val = max(max_val, b);
        max_val = max(max_val, c);
        max_val = max(max_val, d);
    }

    // Process remaining elements
    for (int i = (dim_size / 4) * 4; i < dim_size; ++i) {
        scalar_t val = input[base + i * inner_size];
        max_val = max(max_val, val);
    }

    // Load the computed maximum into shared memory
    sdata[tid] = max_val;
    __syncthreads();

    // Reduce within the block using shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// CUDA forward function with combined optimizations
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

    // Prepare output dimensions by removing the reduced dimension
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty({outer_size, inner_size}, input.options());

    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward_combined", ([&] {
        max_reduce_combined_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));

    // Further reduction in output if necessary
    if (blocks > 1) {
        auto final_output = torch::empty(output_sizes, input.options());
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_final", ([&] {
            max_reduce_combined_kernel<scalar_t><<<1, threads, threads * sizeof(scalar_t)>>>(
                output.data_ptr<scalar_t>(),
                final_output.data_ptr<scalar_t>(),
                1,
                blocks,
                1
            );
        }));
        return final_output;
    }

    return output.view(output_sizes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward combined (CUDA)");
}