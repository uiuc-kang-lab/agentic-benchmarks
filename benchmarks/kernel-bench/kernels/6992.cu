#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <limits>

template <typename scalar_t>
__global__ void argmin_parallel_tuned_kernel(const scalar_t* __restrict__ x,
                                           int64_t* __restrict__ output,
                                           int K,
                                           int64_t inner_size) {
    int outer = blockIdx.y;
    int inner = blockIdx.x * blockDim.y + threadIdx.y;

    if (inner >= inner_size) return;

    // Use block size of 256 threads (8 warps)
    constexpr int BLOCK_SIZE = 256;
    const int tid = threadIdx.x;
    const int WARP_SIZE = 32;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Shared memory declaration
    __shared__ scalar_t s_vals[BLOCK_SIZE];
    __shared__ int s_indices[BLOCK_SIZE];

    const scalar_t* slice_start = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;

    // Initialize with maximum value
    scalar_t local_min = std::numeric_limits<scalar_t>::max();
    int local_idx = 0;

    // Each thread processes multiple elements with stride BLOCK_SIZE
    #pragma unroll 8
    for (int k = tid; k < K; k += BLOCK_SIZE) {
        scalar_t val = slice_start[k * inner_size];
        if (val < local_min) {
            local_min = val;
            local_idx = k;
        }
    }

    // Store local results to shared memory
    s_vals[tid] = local_min;
    s_indices[tid] = local_idx;
    __syncthreads();

    // Warp-level reduction (only first warp participates)
    if (warp_id == 0) {
        // Each thread in the warp reduces 8 elements
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            int idx = tid + i * WARP_SIZE;
            if (idx < BLOCK_SIZE) {
                if (s_vals[idx] < s_vals[tid]) {
                    s_vals[tid] = s_vals[idx];
                    s_indices[tid] = s_indices[idx];
                }
            }
        }

        // Warp reduction using shuffle
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            scalar_t other_val = __shfl_down_sync(0xffffffff, s_vals[tid], offset);
            int other_idx = __shfl_down_sync(0xffffffff, s_indices[tid], offset);
            if (lane_id < offset && other_val < s_vals[tid]) {
                s_vals[tid] = other_val;
                s_indices[tid] = other_idx;
            }
        }

        // First thread writes the result
        if (tid == 0) {
            output[outer * inner_size + inner] = s_indices[0];
        }
    }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

    int dims = x.dim();
    if (dim < 0) dim += dims;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= x.size(i);
    
    int K = static_cast<int>(x.size(dim));
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < dims; i++) inner_size *= x.size(i);

    std::vector<int64_t> out_sizes;
    for (int i = 0; i < dims; i++) {
        if (i != dim) out_sizes.push_back(x.size(i));
    }
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

    // Use block size of 256 threads
    constexpr int BLOCK_SIZE = 256;
    
    // Configure 2D grid for better occupancy
    const int threads_x = BLOCK_SIZE;
    const int threads_y = 1;
    dim3 block_dim(threads_x, threads_y);
    
    const int blocks_x = (inner_size + threads_y - 1) / threads_y;
    dim3 grid_dim(blocks_x, outer_size);

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_parallel_tuned_kernel<scalar_t><<<grid_dim, block_dim>>>(
            x_data, output_data, K, inner_size);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmin_cuda_forward, "Argmin forward with optimized block size (CUDA)");
}