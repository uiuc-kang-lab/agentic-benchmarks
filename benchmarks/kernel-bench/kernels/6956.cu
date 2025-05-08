#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// CUDA kernel using warp shuffle reduction to minimize synchronizations
// and only synchronizing when necessary to combine per-warp results.

template <typename scalar_t>
__global__ void argmin_warp_shuffle_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size) {

    // Each block processes one slice of shape [K] along the reduction dimension.
    int slice_idx = blockIdx.x;
    if (slice_idx >= outer_size * inner_size) return;

    // Decompose slice index into outer and inner indices
    int64_t outer = slice_idx / inner_size;
    int64_t inner = slice_idx % inner_size;

    // Each thread will scan along the K dimension with stride of blockDim.x
    int blockSize = blockDim.x;  // ideally 128
    scalar_t local_min = std::numeric_limits<scalar_t>::max();
    int local_idx = 0;

    for (int k = threadIdx.x; k < K; k += blockSize) {
        scalar_t val = __ldg(&x[outer * (static_cast<int64_t>(K) * inner_size) + k * inner_size + inner]);
        if (val < local_min) {
            local_min = val;
            local_idx = k;
        }
    }

    // Perform warp-level reduction using shuffle instructions
    unsigned int full_mask = 0xffffffff;
    int lane = threadIdx.x % warpSize;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        scalar_t other_min = __shfl_down_sync(full_mask, local_min, offset);
        int other_idx = __shfl_down_sync(full_mask, local_idx, offset);
        if (other_min < local_min) {
            local_min = other_min;
            local_idx = other_idx;
        }
    }

    // Each warp's lane 0 writes its result to shared memory
    __shared__ scalar_t shared_min[32];   // Maximum warps per block if blockDim.x <= 1024
    __shared__ int shared_idx[32];
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) {
        shared_min[warp_id] = local_min;
        shared_idx[warp_id] = local_idx;
    }

    // Synchronize only to ensure the per-warp results are visible for final reduction
    __syncthreads();

    // Final reduction by the first warp using only the active lanes corresponding to the number of warps
    int num_warps = blockDim.x / warpSize;
    if (threadIdx.x < num_warps) {
        local_min = shared_min[threadIdx.x];
        local_idx = shared_idx[threadIdx.x];
        // Create an active mask for the final reduction
        unsigned int final_mask = (1 << num_warps) - 1;
        for (int offset = num_warps / 2; offset > 0; offset /= 2) {
            scalar_t other_min = __shfl_down_sync(final_mask, local_min, offset);
            int other_idx = __shfl_down_sync(final_mask, local_idx, offset);
            if (other_min < local_min) {
                local_min = other_min;
                local_idx = other_idx;
            }
        }
        if (threadIdx.x == 0) {
            output[slice_idx] = local_idx;
        }
    }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    int dims = x.dim();
    if (dim < 0) dim += dims;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

    // Calculate outer_size as the product of dimensions before 'dim'
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }
    // K is the size of the reduction dimension
    int K = static_cast<int>(x.size(dim));
    // Calculate inner_size as the product of dimensions after 'dim'
    int64_t inner_size = 1;
    for (int i = dim + 1; i < dims; i++) {
        inner_size *= x.size(i);
    }

    // Determine output tensor shape (same as input except the reduced dim is removed)
    std::vector<int64_t> out_sizes;
    for (int i = 0; i < dims; i++) {
        if (i == dim) continue;
        out_sizes.push_back(x.size(i));
    }
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

    // Launch one block per slice; using 128 threads per block (which works nicely with warp shuffle reduction)
    int threads = 128;
    int blocks = outer_size * inner_size;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_warp_shuffle_kernel<scalar_t><<<blocks, threads>>>(
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
