#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel that leverages shared memory for intra-block reduction
// and warp-level primitives (__shfl_down_sync) for the final reduction stage.
// Each block computes one output element corresponding to the reduced dimension.

template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {

    extern __shared__ scalar_t shared_data[];  // Shared memory for storing partial sums

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;

    // Each block corresponds to one output element
    const int out_idx = blockIdx.x;
    if (out_idx >= outer_size * inner_size)
        return;

    // Determine the indices for outer and inner dimensions
    const int outer_idx = out_idx / inner_size;
    const int inner_idx = out_idx % inner_size;
    const int base_idx = outer_idx * dim_size * inner_size + inner_idx;

    // Each thread computes a partial sum over the reduction dimension using a grid-stride loop
    scalar_t thread_sum = 0;
    for (int i = tid; i < dim_size; i += blockSize) {
        thread_sum += input[base_idx + i * inner_size];
    }

    // Store the partial sum into shared memory
    shared_data[tid] = thread_sum;
    __syncthreads();

    // Intra-block reduction using shared memory until only 32 elements remain
    for (int stride = blockSize / 2; stride > 32; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Final reduction using warp-level primitives without additional synchronization
    if (tid < 32) {
        // Load the final values; using volatile ensures the compiler doesn't optimize out needed loads.
        volatile scalar_t val = shared_data[tid];
        // Unrolled warp-level reduction using __shfl_down_sync
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) {
            // Write the mean result to global memory (division done once here)
            output[out_idx] = val / static_cast<scalar_t>(dim_size);
        }
    }
}

// Host function to prepare and launch the CUDA kernel

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    // Compute the sizes of the input tensor
    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }

    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Remove the reduced dimension for the output shape
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    // Kernel launch configuration
    const int threads = 256;
    const int blocks = outer_size * inner_size;
    // Shared memory size in bytes = number of threads * size of scalar_t
    const int shared_mem_size = threads * input.element_size();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Mean reduction optimized with shared memory and warp primitives (CUDA)");
}
