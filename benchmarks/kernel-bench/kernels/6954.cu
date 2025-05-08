#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <limits>

// This kernel assigns one warp per slice (i.e., each combination of outer and inner indices).
// Each warp processes the reduction dimension (of size K) using warp-level primitives to do the reduction,
// ensuring that workloads are evenly distributed among threads and blocks. Using __ldg helps optimize global memory loads.

template <typename scalar_t>
__global__ void argmin_warp_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t total_slices,
    int64_t inner_size) {

    const int warpSize = 32;
    // Each warp is identified by its global warp id
    int global_warp_id = (blockIdx.x * (blockDim.x / warpSize)) + (threadIdx.x / warpSize);
    if (global_warp_id >= total_slices) return;

    int lane = threadIdx.x % warpSize;  // lane within the warp
    int64_t slice = global_warp_id;      // each warp processes one slice
    int64_t outer = slice / inner_size;
    int64_t inner = slice % inner_size;

    scalar_t min_val = std::numeric_limits<scalar_t>::max();
    int min_index = 0;

    // Distribute the workload of processing K elements across the 32 threads in the warp
    for (int k = lane; k < K; k += warpSize) {
        // Use __ldg for read-only data loading
        scalar_t val = __ldg(&x[ outer * ((int64_t)K * inner_size) + k * inner_size + inner ]);
        if (val < min_val) {
            min_val = val;
            min_index = k;
        }
    }

    // Warp-level reduction using shuffle intrinsics to combine results from all lanes
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        scalar_t other_val = __shfl_down_sync(0xffffffff, min_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, min_index, offset);
        if (other_val < min_val) {
            min_val = other_val;
            min_index = other_idx;
        }
    }

    // The first lane writes the final result
    if (lane == 0) {
        output[slice] = min_index;
    }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

    int dims = x.dim();
    if (dim < 0) dim += dims;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

    // Compute outer_size = product of dimensions before 'dim'
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }
    
    // K is the size of the reduction dimension
    int K = static_cast<int>(x.size(dim));

    // Compute inner_size = product of dimensions after 'dim'
    int64_t inner_size = 1;
    for (int i = dim + 1; i < dims; i++) {
        inner_size *= x.size(i);
    }
    
    // Prepare the output tensor with the reduction dimension removed
    std::vector<int64_t> out_sizes;
    for (int i = 0; i < dims; i++) {
        if (i == dim) continue;
        out_sizes.push_back(x.size(i));
    }
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

    // Total slices to reduce
    int64_t total_slices = outer_size * inner_size;

    // Configure kernel launch: use 256 threads per block (i.e., 8 warps per block)
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int num_blocks = (total_slices + warps_per_block - 1) / warps_per_block;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_warp_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            x_data, output_data, K, total_slices, inner_size);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmin_cuda_forward, "Argmin forward (CUDA)");
}
