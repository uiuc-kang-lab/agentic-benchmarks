#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <vector>

// Combined CUDA kernel: dynamic block size selection + warp shuffle reduction

template <typename scalar_t, int BLOCK_SIZE>
__global__ void argmin_combined_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size) {

    const int tid = threadIdx.x;
    const int slice_idx = blockIdx.x;  // each block processes one slice
    if (slice_idx >= outer_size * inner_size) return;

    // Decompose slice index into outer and inner indices
    int64_t outer = slice_idx / inner_size;
    int64_t inner = slice_idx % inner_size;

    // Initialize local minimum
    scalar_t local_min = std::numeric_limits<scalar_t>::max();
    int local_idx = 0;

    // Each thread processes a subset of elements along the reduction dimension
    for (int k = tid; k < K; k += BLOCK_SIZE) {
        // Use __ldg for cached memory read
        scalar_t val = __ldg(&x[outer * (static_cast<int64_t>(K) * inner_size) + k * inner_size + inner]);
        if (val < local_min) {
            local_min = val;
            local_idx = k;
        }
    }

    // Allocate shared memory statically
    __shared__ scalar_t s_min_vals[BLOCK_SIZE];
    __shared__ int s_min_inds[BLOCK_SIZE];

    s_min_vals[tid] = local_min;
    s_min_inds[tid] = local_idx;
    __syncthreads();

    // Perform reduction in shared memory until we reach warp-level
    for (int stride = BLOCK_SIZE / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            if (s_min_vals[tid + stride] < s_min_vals[tid]) {
                s_min_vals[tid] = s_min_vals[tid + stride];
                s_min_inds[tid] = s_min_inds[tid + stride];
            }
        }
        __syncthreads();
    }

    // Warp-level reduction using shuffle down for lower latency
    if (tid < 32) {
        // Load the reduced value from shared memory into registers
        scalar_t val = s_min_vals[tid];
        int idx = s_min_inds[tid];
        // Unroll reduction using warp shuffle
        for (int offset = 16; offset > 0; offset /= 2) {
            scalar_t other_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
            int other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
            if (other_val < val) {
                val = other_val;
                idx = other_idx;
            }
        }
        if (tid == 0) {
            output[slice_idx] = idx;
        }
    }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

    int dims = x.dim();
    if (dim < 0) dim += dims;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

    // Compute outer_size (product of dims before 'dim')
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }
    // K is the size of the reduction dimension
    int K = static_cast<int>(x.size(dim));
    // Compute inner_size (product of dims after 'dim')
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

    // Dynamically choose an optimal block size based on K
    int block_size;
    if (K <= 32) {
        block_size = 32;
    } else if (K <= 64) {
        block_size = 64;
    } else if (K <= 128) {
        block_size = 128;
    } else if (K <= 256) {
        block_size = 256;
    } else if (K <= 512) {
        block_size = 512;
    } else {
        block_size = 1024;
    }

    int num_slices = outer_size * inner_size;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        switch (block_size) {
            case 32:
                argmin_combined_kernel<scalar_t, 32><<<num_slices, 32>>>(
                    x_data, output_data, K, outer_size, inner_size);
                break;
            case 64:
                argmin_combined_kernel<scalar_t, 64><<<num_slices, 64>>>(
                    x_data, output_data, K, outer_size, inner_size);
                break;
            case 128:
                argmin_combined_kernel<scalar_t, 128><<<num_slices, 128>>>(
                    x_data, output_data, K, outer_size, inner_size);
                break;
            case 256:
                argmin_combined_kernel<scalar_t, 256><<<num_slices, 256>>>(
                    x_data, output_data, K, outer_size, inner_size);
                break;
            case 512:
                argmin_combined_kernel<scalar_t, 512><<<num_slices, 512>>>(
                    x_data, output_data, K, outer_size, inner_size);
                break;
            case 1024:
                argmin_combined_kernel<scalar_t, 1024><<<num_slices, 1024>>>(
                    x_data, output_data, K, outer_size, inner_size);
                break;
            default:
                argmin_combined_kernel<scalar_t, 128><<<num_slices, 128>>>(
                    x_data, output_data, K, outer_size, inner_size);
                break;
        }
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmin_cuda_forward, "Argmin forward (CUDA) - Combined Optimized Kernel");
}
