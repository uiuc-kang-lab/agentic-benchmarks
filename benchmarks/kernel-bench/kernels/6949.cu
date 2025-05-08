#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <limits>

// Define warp size
#define WARP_SIZE 32

// This kernel uses a templated block size and combines dynamic block selection with warp-level reduction
// using shuffle intrinsics to decrease shared memory usage and synchronization overhead.

template <typename scalar_t, int BLOCK_SIZE>
__global__ void combined_argmin_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size) {

    const int tid = threadIdx.x;
    const int block_size = BLOCK_SIZE;
    const int lane = tid % WARP_SIZE;
    const int warpId = tid / WARP_SIZE;

    // Each block processes one slice
    int64_t slice_idx = blockIdx.x;
    if (slice_idx >= outer_size * inner_size) return;

    // Decompose slice index into outer and inner indices
    int64_t outer = slice_idx / inner_size;
    int64_t inner = slice_idx % inner_size;

    // Initialize local minimum and its index
    scalar_t local_min = std::numeric_limits<scalar_t>::max();
    int local_min_idx = 0;

    // Each thread processes elements along the reduction dimension with stride = block_size
    for (int k = tid; k < K; k += block_size) {
        scalar_t val = __ldg(&x[ outer * (static_cast<int64_t>(K) * inner_size) + k * inner_size + inner ]);
        if (val < local_min) {
            local_min = val;
            local_min_idx = k;
        }
    }

    // Intra-warp reduction using warp shuffle
    unsigned mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        scalar_t temp = __shfl_down_sync(mask, local_min, offset);
        int temp_idx = __shfl_down_sync(mask, local_min_idx, offset);
        if (temp < local_min) {
            local_min = temp;
            local_min_idx = temp_idx;
        }
    }

    // Allocate shared memory for storing one candidate per warp.
    // We reserve BLOCK_SIZE elements for min values and BLOCK_SIZE for indices (only first (block_size/WARP_SIZE) are used).
    extern __shared__ char smem[];
    scalar_t* s_min_vals = reinterpret_cast<scalar_t*>(smem);
    int* s_min_inds = reinterpret_cast<int*>(smem + block_size * sizeof(scalar_t));

    // The first lane of each warp writes its candidate to shared memory
    if (lane == 0) {
        s_min_vals[warpId] = local_min;
        s_min_inds[warpId] = local_min_idx;
    }
    __syncthreads();

    // Final reduction among warp candidates is done by thread 0 of the block
    if (tid == 0) {
        int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;
        scalar_t final_min = s_min_vals[0];
        int final_idx = s_min_inds[0];
        for (int i = 1; i < num_warps; i++) {
            scalar_t candidate = s_min_vals[i];
            int candidate_idx = s_min_inds[i];
            if (candidate < final_min) {
                final_min = candidate;
                final_idx = candidate_idx;
            }
        }
        output[slice_idx] = final_idx;
    }
}


at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

    int dims = x.dim();
    if (dim < 0) dim += dims;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

    // Compute outer_size (product of dimensions before 'dim')
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }
    // K is the size of the reduction dimension
    int K = static_cast<int>(x.size(dim));
    // Compute inner_size (product of dimensions after 'dim')
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

    // Dynamically choose optimal block size based on K
    int block_size;
    if (K <= 32) {
        block_size = 32;
    } else if (K <= 64) {
        block_size = 64;
    } else if (K <= 128) {
        block_size = 128;
    } else if (K <= 256) {
        block_size = 256;
    } else {
        block_size = 512;
    }

    int num_slices = outer_size * inner_size;

    // Dispatch the kernel with dynamic shared memory size:
    // Shared memory size = block_size * sizeof(scalar_t) + block_size * sizeof(int)
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "combined_argmin_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        switch (block_size) {
            case 32:
                combined_argmin_kernel<scalar_t, 32><<<num_slices, 32, 32 * sizeof(scalar_t) + 32 * sizeof(int)>>>(
                    x_data, output_data, K, outer_size, inner_size);
                break;
            case 64:
                combined_argmin_kernel<scalar_t, 64><<<num_slices, 64, 64 * sizeof(scalar_t) + 64 * sizeof(int)>>>(
                    x_data, output_data, K, outer_size, inner_size);
                break;
            case 128:
                combined_argmin_kernel<scalar_t, 128><<<num_slices, 128, 128 * sizeof(scalar_t) + 128 * sizeof(int)>>>(
                    x_data, output_data, K, outer_size, inner_size);
                break;
            case 256:
                combined_argmin_kernel<scalar_t, 256><<<num_slices, 256, 256 * sizeof(scalar_t) + 256 * sizeof(int)>>>(
                    x_data, output_data, K, outer_size, inner_size);
                break;
            case 512:
                combined_argmin_kernel<scalar_t, 512><<<num_slices, 512, 512 * sizeof(scalar_t) + 512 * sizeof(int)>>>(
                    x_data, output_data, K, outer_size, inner_size);
                break;
            default:
                combined_argmin_kernel<scalar_t, 256><<<num_slices, 256, 256 * sizeof(scalar_t) + 256 * sizeof(int)>>>(
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
    m.def("forward", &argmin_cuda_forward, "Combined Argmin forward (CUDA) with warp shuffle reduction");
}
