#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <limits>

// Kernel that minimizes __syncthreads() usage by synchronizing only when necessary.
// Reduction across threads > 32 is done with __syncthreads(), then warp-level reduction is done without extra synchronization using volatile pointers.

template <typename scalar_t>
__global__ void argmin_few_sync_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size) {

    int tid = threadIdx.x;
    int block_size = blockDim.x;  // expecting 128 threads
    int64_t slice_idx = blockIdx.x;
    if (slice_idx >= outer_size * inner_size) return;

    int64_t outer = slice_idx / inner_size;
    int64_t inner = slice_idx % inner_size;

    // Shared memory buffers sized for 128 threads
    __shared__ scalar_t s_min_vals[128];
    __shared__ int s_min_indices[128];

    scalar_t local_min = std::numeric_limits<scalar_t>::max();
    int local_min_idx = 0;

    // Each thread iterates over the reduction dimension in strides of block_size
    for (int k = tid; k < K; k += block_size) {
        scalar_t val = __ldg(&x[outer * (static_cast<int64_t>(K) * inner_size) + k * inner_size + inner]);
        if (val < local_min) {
            local_min = val;
            local_min_idx = k;
        }
    }

    // Write results into shared memory
    s_min_vals[tid] = local_min;
    s_min_indices[tid] = local_min_idx;

    // Synchronize to ensure all threads have written their partial results
    __syncthreads();

    // Reduction: loop until only 32 threads remain
    for (int s = block_size; s > 32; s >>= 1) {
        if (tid < (s >> 1)) {
            scalar_t other = s_min_vals[tid + (s >> 1)];
            if (other < s_min_vals[tid]) {
                s_min_vals[tid] = other;
                s_min_indices[tid] = s_min_indices[tid + (s >> 1)];
            }
        }
        __syncthreads();
    }

    // Final warp-level reduction without __syncthreads(), relying on warp-synchronous execution
    if (tid < 32) {
        volatile scalar_t* vs_min_vals = s_min_vals;
        volatile int* vs_min_indices = s_min_indices;
        if (vs_min_vals[tid + 16] < vs_min_vals[tid]) {
            vs_min_vals[tid] = vs_min_vals[tid + 16];
            vs_min_indices[tid] = vs_min_indices[tid + 16];
        }
        if (vs_min_vals[tid + 8] < vs_min_vals[tid]) {
            vs_min_vals[tid] = vs_min_vals[tid + 8];
            vs_min_indices[tid] = vs_min_indices[tid + 8];
        }
        if (vs_min_vals[tid + 4] < vs_min_vals[tid]) {
            vs_min_vals[tid] = vs_min_vals[tid + 4];
            vs_min_indices[tid] = vs_min_indices[tid + 4];
        }
        if (vs_min_vals[tid + 2] < vs_min_vals[tid]) {
            vs_min_vals[tid] = vs_min_vals[tid + 2];
            vs_min_indices[tid] = vs_min_indices[tid + 2];
        }
        if (vs_min_vals[tid + 1] < vs_min_vals[tid]) {
            vs_min_vals[tid] = vs_min_vals[tid + 1];
            vs_min_indices[tid] = vs_min_indices[tid + 1];
        }
    }

    if (tid == 0) {
        output[slice_idx] = s_min_indices[0];
    }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

    int dims = x.dim();
    if (dim < 0) dim += dims;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

    // Calculate outer_size and inner_size based on the reduction dimension
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

    // Launch kernel with 128 threads per block
    int threads = 128;
    int blocks = outer_size * inner_size;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_few_sync_kernel<scalar_t><<<blocks, threads>>>(
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
