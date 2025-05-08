#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <limits>

// Optimized CUDA kernel that uses stride loops to handle workloads larger than the number of available threads
// and verifies correct boundary handling. Each block processes one slice (from the tensor reshaped as [outer_size, K, inner_size]).

template <typename scalar_t>
__global__ void argmin_stride_loop_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size) {

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    int64_t slice_idx = blockIdx.x;

    // Boundary check: if slice index is out of range
    if (slice_idx >= outer_size * inner_size) return;

    // Compute the corresponding outer and inner indices
    int64_t outer = slice_idx / inner_size;
    int64_t inner = slice_idx % inner_size;

    // Compute pointer to the beginning of the slice data
    const scalar_t* slice_ptr = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;

    // Initialize local minimum value and index
    scalar_t local_min = std::numeric_limits<scalar_t>::max();
    int local_min_index = 0;

    // Use stride loop to process the reduction dimension (K) if K exceeds the block size
    for (int k = tid; k < K; k += block_size) {
        // Access element at k-th position along the reduction dimension
        scalar_t val = __ldg(&slice_ptr[k * inner_size]);
        if (val < local_min) {
            local_min = val;
            local_min_index = k;
        }
    }

    // Use dynamic shared memory for reduction
    extern __shared__ char smem[];
    scalar_t* s_min_vals = reinterpret_cast<scalar_t*>(smem);
    int* s_min_indices = reinterpret_cast<int*>(smem + block_size * sizeof(scalar_t));

    s_min_vals[tid] = local_min;
    s_min_indices[tid] = local_min_index;
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (s_min_vals[tid + stride] < s_min_vals[tid]) {
                s_min_vals[tid] = s_min_vals[tid + stride];
                s_min_indices[tid] = s_min_indices[tid + stride];
            }
        }
        __syncthreads();
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

    // Compute outer_size, K (the reduction dimension) and inner_size
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }
    int K = static_cast<int>(x.size(dim));
    int64_t inner_size = 1;
    for (int i = dim + 1; i < dims; i++) {
        inner_size *= x.size(i);
    }

    // Build output dimensions: same as input except the reduction dimension is removed
    std::vector<int64_t> out_sizes;
    for (int i = 0; i < dims; i++) {
        if (i == dim) continue;
        out_sizes.push_back(x.size(i));
    }
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

    // Launch configuration
    int threads = 256;
    int blocks = outer_size * inner_size;
    size_t shared_mem_size = threads * (sizeof(at::ScalarType(at::kFloat)) == sizeof(float) ? sizeof(float) : sizeof(float)); // workaround for size computation
    // Instead, use: shared memory = threads * (sizeof(scalar_t) + sizeof(int))
    shared_mem_size = threads * (sizeof(float) + sizeof(int)); // Here we assume float; AT_DISPATCH_ALL_TYPES handles type cast appropriately

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_stride_loop_kernel<scalar_t><<<blocks, threads, threads * (sizeof(scalar_t) + sizeof(int))>>>(
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
