#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel for mean reduction combining warp shuffle and unrolled shared memory reduction

template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size) {

    // Allocate shared memory to store per-warp sums
    extern __shared__ char shared_mem[];
    scalar_t* warp_sums = reinterpret_cast<scalar_t*>(shared_mem);

    const int tid = threadIdx.x;
    const int lane = tid & 31;        // Thread index within warp
    const int warpId = tid >> 5;      // Warp index within the block

    // Each block computes one output element
    const int output_idx = blockIdx.x;
    if (output_idx >= outer_size * inner_size) {
        return;
    }

    // Calculate the corresponding indices for the reduction dimension
    const int outer_idx = output_idx / inner_size;
    const int inner_idx = output_idx % inner_size;
    const int base_idx = outer_idx * dim_size * inner_size + inner_idx;

    // Each thread computes a partial sum over the reduction dimension
    scalar_t thread_sum = 0;
    // Use unrolling hint for improved performance
    #pragma unroll
    for (int i = tid; i < dim_size; i += blockDim.x) {
        thread_sum += input[base_idx + i * inner_size];
    }

    // Perform efficient warp-level reduction with shuffle
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // The first thread in each warp writes its sum to shared memory
    if (lane == 0) {
        warp_sums[warpId] = thread_sum;
    }
    __syncthreads();

    // Now, reduce the sums from each warp.
    // The number of warps per block is blockDim.x/32
    int num_warps = blockDim.x >> 5;
    if (tid < num_warps) {
        scalar_t sum = warp_sums[tid];
        // Use warp shuffle to reduce the warp sums efficiently
        for (int offset = num_warps >> 1; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        // The first thread in the block writes the final output
        if (tid == 0) {
            output[output_idx] = sum / static_cast<scalar_t>(dim_size);
        }
    }
}

// Host function to prepare and launch the optimized CUDA kernel

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Adjust negative dimension values
    if (dim < 0) dim += input.dim();

    // Calculate sizes and shapes
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

    // Remove the reduced dimension from output shape
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    // Kernel launch configuration
    const int threads = 256;
    const int blocks = outer_size * inner_size;
    // Allocate shared memory: one scalar per warp
    const int shared_mem_size = (threads >> 5) * input.element_size();

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
    m.def("forward", &mean_reduce_cuda, "Mean reduction optimized (CUDA)");
}
