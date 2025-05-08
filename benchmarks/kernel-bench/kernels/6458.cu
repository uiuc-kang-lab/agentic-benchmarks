#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <type_traits>

// Define block configuration
#define BLOCK_SIZE 256
#define ITEMS_PER_THREAD 4

// Kernel for single-block reduction (no atomic), used when the reduction dimension is small
// Each block processes one output element
// Leverage shared memory for reduction
template <typename scalar_t>
__global__ void mean_reduce_kernel_shared(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int L,           // reduction dimension length
    int stride,      // stride for accessing reduction elements
    int N            // number of output elements
) {
    int out_idx = blockIdx.x;  // each block handles one output element (flattened)
    if (out_idx >= N) return;
    
    // Decode the flat output index into outer and inner indices
    int outer_idx = out_idx / stride;
    int inner_idx = out_idx % stride;
    // Base offset for the current output element's reduction slice in the input
    int base_offset = outer_idx * (L * stride) + inner_idx;

    __shared__ scalar_t sdata[BLOCK_SIZE];
    scalar_t sum = static_cast<scalar_t>(0);

    // Each thread accumulates over a strided loop
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
         sum += __ldg(input + base_offset + i * stride);
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
         if (threadIdx.x < s)
              sdata[threadIdx.x] += sdata[threadIdx.x + s];
         __syncthreads();
    }
    
    if (threadIdx.x == 0) {
         // Write the final mean directly (no atomic needed)
         output[out_idx] = sdata[0] / static_cast<scalar_t>(L);
    }
}

// Host function that selects the appropriate kernel launch based on reduction dimension size
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    // Get input sizes and compute L, outer_size, and inner_size
    std::vector<int64_t> sizes = input.sizes().vec();
    int64_t L = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
         outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < sizes.size(); i++) {
         inner_size *= sizes[i];
    }
    // Total number of output elements
    int64_t N = outer_size * inner_size;
    int stride = inner_size;  // For computing input index: reduction elements are spaced by inner_size

    torch::Tensor output = torch::empty({N}, input.options());
    int blocks = N;
    int threads = BLOCK_SIZE;
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda_shared", ([&] {
         mean_reduce_kernel_shared<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<int>(L),
            stride,
            static_cast<int>(N)
         );
    }));

    // Reshape output to remove the reduced dimension
    sizes.erase(sizes.begin() + dim);
    output = output.view(sizes);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Mean Reduction (Shared Memory Optimized CUDA)");
}
