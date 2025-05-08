#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Kernel that minimizes warp divergence by using warp shuffle based reduction with uniform control flow.

template <typename scalar_t>
__global__ void warp_shfl_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t num_outputs
) {
    // Define warp size and compute number of warps per block
    const int warpSize = 32;
    extern __shared__ char shared[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(shared);

    // Each block processes one or more output elements using a grid-stride loop
    for (int out_idx = blockIdx.x; out_idx < num_outputs; out_idx += gridDim.x) {
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;
        // Compute the starting index in the input tensor for the current reduction
        int64_t base = outer_idx * dim_size * inner_size + inner_idx;

        // Initialize each thread's local maximum with the lowest possible value
        scalar_t thread_max = std::numeric_limits<scalar_t>::lowest();

        // Each thread processes a subset of the reduction dimension in a uniform grid-stride loop
        for (int j = threadIdx.x; j < dim_size; j += blockDim.x) {
            scalar_t val = input[base + j * inner_size];
            thread_max = max(thread_max, val);
        }

        // Perform warp-level reduction using shuffle operations to avoid divergent branching
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            // All threads in the warp uniformly participate in the reduction
            scalar_t other = __shfl_down_sync(0xffffffff, thread_max, offset);
            thread_max = max(thread_max, other);
        }

        // Write each warp's result into shared memory. Use the first lane of each warp.
        int lane = threadIdx.x & (warpSize - 1);
        int warp_id = threadIdx.x / warpSize;
        // Compute number of warps per block
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        if (lane == 0) {
            sdata[warp_id] = thread_max;
        }
        __syncthreads();

        // Let the first warp perform reduction on the results from each warp with uniform control flow
        if (threadIdx.x < num_warps) {
            scalar_t block_max = sdata[threadIdx.x];
            for (int offset = warpSize/2; offset > 0; offset /= 2) {
                scalar_t other = __shfl_down_sync(0xffffffff, block_max, offset);
                block_max = max(block_max, other);
            }
            if (threadIdx.x == 0) {
                output[out_idx] = block_max;
            }
        }
        __syncthreads(); // Ensure all threads are ready for the next output element
    }
}

// CUDA forward function
torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    // Handle negative dimension
    if (dim < 0) dim += input.dim();

    // Calculate outer and inner sizes based on the reduction dimension
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

    // Prepare output tensor with the reduced dimension removed
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // Set kernel configuration
    int threads = 256;
    int blocks = (num_outputs < 1024 ? num_outputs : 1024);
    // Allocate shared memory: one scalar per warp
    const int warpSize = 32;
    int num_warps = (threads + warpSize - 1) / warpSize;
    size_t shared_mem_size = num_warps * input.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "warp_shfl_reduce_forward", ([&] {
        warp_shfl_reduce_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward (CUDA) with warp shuffle reduction");
}
