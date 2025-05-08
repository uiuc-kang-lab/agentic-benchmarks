#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <limits>

// This kernel uses stride loops so that each thread handles multiple elements when K exceeds blockDim.x.
// It then employs warp-level shuffle reductions to quickly reduce within each warp, and finally uses shared memory
// to combine results across warps. Boundary conditions are verified by ensuring that each block handles one slice
// of the tensor (after reshaping to [outer_size, K, inner_size]).

template <typename scalar_t>
__global__ void argmin_stride_loop_warp_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size) {

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    int64_t slice = blockIdx.x;
    if (slice >= outer_size * inner_size) return;

    // Compute the outer and inner indices to locate the slice
    int64_t outer = slice / inner_size;
    int64_t inner = slice % inner_size;
    const scalar_t* slice_ptr = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;

    // Initialize local minimum value and associated index
    scalar_t local_min = std::numeric_limits<scalar_t>::max();
    int local_idx = 0;

    // Use a stride loop to cover all elements along the reduction dimension
    for (int k = tid; k < K; k += block_size) {
        scalar_t val = __ldg(&slice_ptr[k * inner_size]);
        if (val < local_min) {
            local_min = val;
            local_idx = k;
        }
    }

    // Warp-level reduction using shuffle down
    unsigned int mask = 0xFFFFFFFF;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        scalar_t other_val = __shfl_down_sync(mask, local_min, offset);
        int other_idx = __shfl_down_sync(mask, local_idx, offset);
        if (other_val < local_min) {
            local_min = other_val;
            local_idx = other_idx;
        }
    }

    // Each warp's first thread writes its result to shared memory
    __shared__ scalar_t s_warp_min[32];
    __shared__ int s_warp_idx[32];
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) {
        s_warp_min[warp_id] = local_min;
        s_warp_idx[warp_id] = local_idx;
    }
    __syncthreads();

    // Final reduction: let the first few threads (one per warp) reduce the warp results
    int num_warps = (block_size + warpSize - 1) / warpSize;
    if (tid < num_warps) {
        local_min = s_warp_min[lane];
        local_idx = s_warp_idx[lane];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            scalar_t other_val = __shfl_down_sync(mask, local_min, offset);
            int other_idx = __shfl_down_sync(mask, local_idx, offset);
            if (other_val < local_min) {
                local_min = other_val;
                local_idx = other_idx;
            }
        }
        if (lane == 0)
            s_warp_idx[0] = local_idx;
    }
    __syncthreads();

    // Thread 0 writes the final argmin for this slice
    if (tid == 0) {
        output[slice] = s_warp_idx[0];
    }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    int dims = x.dim();
    if (dim < 0) dim += dims;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

    // Compute outer_size, reduction dimension size (K), and inner_size
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }
    int K = static_cast<int>(x.size(dim));
    int64_t inner_size = 1;
    for (int i = dim + 1; i < dims; i++) {
        inner_size *= x.size(i);
    }

    // Prepare output tensor with the reduction dimension removed
    std::vector<int64_t> out_sizes;
    for (int i = 0; i < dims; i++) {
        if (i == dim) continue;
        out_sizes.push_back(x.size(i));
    }
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

    int threads = 256;
    int blocks = outer_size * inner_size;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_stride_loop_warp_kernel<scalar_t><<<blocks, threads>>>(
            x_data, output_data, K, outer_size, inner_size);
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
