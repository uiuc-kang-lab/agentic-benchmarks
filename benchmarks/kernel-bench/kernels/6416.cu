#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


// Kernel using warp-level primitives (__shfl_down_sync) for reduction
template <typename scalar_t>
__global__ void warp_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size) {

    const int WARP_SIZE = 32;
    const unsigned int FULL_MASK = 0xffffffffu;

    // Each block computes one output element
    int out_idx = blockIdx.x;  // flattened index over outer * inner dimensions
    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;

    // Base index in input for this reduction segment
    int64_t base = outer_idx * reduce_size * inner_size + inner_idx;

    // Each thread computes a partial sum over a strided portion of the reduction dimension
    scalar_t sum = 0;
    for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        sum += input[base + i * inner_size];
    }

    // Reduce within each warp using warp shuffle intrinsics
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    // Lane index and warp id
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x / WARP_SIZE;

    // Allocate shared memory for per-warp partial sums
    __shared__ scalar_t sWarpSums[32];  // Maximum 32 warps per block
    if (lane == 0) {
        sWarpSums[warpId] = sum;
    }
    __syncthreads();

    // Let the first warp reduce the per-warp sums
    int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    if (threadIdx.x < numWarps) {
        scalar_t warp_sum = sWarpSums[threadIdx.x];
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(FULL_MASK, warp_sum, offset);
        }
        if (threadIdx.x == 0) {
            output[out_idx] = warp_sum;
        }
    }
}


// Host function wrapping the warp-level reduction kernel
torch::Tensor sum_reduce_warp_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    // Compute reduction sizes: outer, reduce, and inner dimensions
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

    // Set output shape: same as input but with reduction dimension set to 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());
    
    // Total number of output elements
    int64_t num_out = outer_size * inner_size;

    // Choose number of threads per block
    int threads = 256;  // can be tuned based on the input size
    // Calculate number of warps per block
    int numWarps = (threads + 31) / 32;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_warp_cuda", ([&] {
        size_t shared_bytes = numWarps * sizeof(scalar_t);
        // Launch one block per output element
        warp_reduce_kernel<scalar_t><<<num_out, threads, shared_bytes>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_warp_cuda, "Sum reduction with warp-level primitives (CUDA)");
}
