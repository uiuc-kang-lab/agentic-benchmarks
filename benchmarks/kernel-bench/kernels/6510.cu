#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Warp-level reduction using cooperative groups
template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val, cg::thread_block_tile<32>& tile) {
    #pragma unroll
    for (int offset = tile.size()/2; offset > 0; offset /= 2) {
        val += tile.shfl_down(val, offset);
    }
    return val;
}

// Kernel that computes the mean reduction over a dimension using unrolled loops
template <typename scalar_t>
__global__ void mean_reduce_unroll_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size) {

    extern __shared__ char shared_mem[];
    scalar_t* warp_sums = reinterpret_cast<scalar_t*>(shared_mem);

    const int tid = threadIdx.x;
    const int lane = tid & 31; // thread index within warp
    const int warpId = tid >> 5; // warp index within block

    // Each block computes one output element
    const int output_idx = blockIdx.x;
    if (output_idx >= outer_size * inner_size)
        return;

    // Calculate indices corresponding to the reduction dimension
    const int outer_idx = output_idx / inner_size;
    const int inner_idx = output_idx % inner_size;
    const int base_idx = outer_idx * dim_size * inner_size + inner_idx;

    // Each thread computes a partial sum over the reduction dimension
    scalar_t thread_sum = 0;
    // Unroll the loop to reduce overhead
    #pragma unroll
    for (int i = tid; i < dim_size; i += blockDim.x) {
        thread_sum += input[base_idx + i * inner_size];
    }

    // Perform warp-level reduction using shuffle with unrolling
    thread_sum = warp_reduce_sum(thread_sum);

    // Write each warp's result to shared memory
    if (lane == 0) {
        warp_sums[warpId] = thread_sum;
    }
    __syncthreads();

    // Final reduction of warp results performed by the first warp
    int num_warps = blockDim.x >> 5;
    if (tid < num_warps) {
        scalar_t sum = warp_sums[tid];
        #pragma unroll
        for (int offset = num_warps >> 1; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            output[output_idx] = sum / static_cast<scalar_t>(dim_size);
        }
    }
}

// Host function to launch the kernel

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Adjust negative dimension if needed
    if (dim < 0) dim += input.dim();

    // Get input sizes
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

    // Prepare output shape by removing the reduced dimension
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    // Launch configuration
    const int threads = 256;
    const int blocks = outer_size * inner_size;
    // Shared memory: one scalar per warp
    const int shared_mem_size = (threads >> 5) * input.element_size();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_unroll_cuda", ([&] {
        mean_reduce_unroll_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &mean_reduce_cuda, "Mean reduction with loop unrolling (CUDA)");
}
