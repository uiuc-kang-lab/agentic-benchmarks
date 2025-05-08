#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

// Helper device function to get the global thread index
__device__ inline int get_global_index() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

// Device function for contiguous reduction using vectorized loads
// Attempts vectorized loads for float and double when possible
template <typename scalar_t>
__device__ inline scalar_t reduce_contiguous(const scalar_t* data, int dim_size) {
    scalar_t sum = static_cast<scalar_t>(0);
    if constexpr (std::is_same<scalar_t, float>::value) {
        // Use float4 if 128-bit aligned and number of elements is a multiple of 4
        if ((dim_size % 4 == 0) && ((((uintptr_t)data) & 0xF) == 0)) {
            int vec_iters = dim_size / 4;
            #pragma unroll
            for (int i = 0; i < vec_iters; i++) {
                float4 vals = __ldg(reinterpret_cast<const float4*>(data) + i);
                sum += vals.x + vals.y + vals.z + vals.w;
            }
            return sum;
        }
    } else if constexpr (std::is_same<scalar_t, double>::value) {
        // Use double2 if 128-bit aligned and number of elements is a multiple of 2
        if ((dim_size % 2 == 0) && ((((uintptr_t)data) & 0xF) == 0)) {
            int vec_iters = dim_size / 2;
            #pragma unroll
            for (int i = 0; i < vec_iters; i++) {
                double2 vals = __ldg(reinterpret_cast<const double2*>(data) + i);
                sum += vals.x + vals.y;
            }
            return sum;
        }
    }
    // Fallback to scalar loads when vectorized load is not applicable
    #pragma unroll
    for (int i = 0; i < dim_size; i++) {
         sum += __ldg(data + i);
    }
    return sum;
}

// Device function for strided (non-contiguous) reduction
// Loads elements from memory that are offset by the given stride
template <typename scalar_t>
__device__ inline scalar_t reduce_strided(const scalar_t* data, int dim_size, int stride) {
    scalar_t sum = static_cast<scalar_t>(0);
    #pragma unroll
    for (int i = 0; i < dim_size; i++) {
         sum += __ldg(data + i * stride);
    }
    return sum;
}

// Unified reduction function that selects the appropriate reduction strategy
// based on the contiguity of the reduction dimension
template <typename scalar_t>
__device__ inline scalar_t reduce_sum(const scalar_t* data, int dim_size, int inner_size) {
    if (inner_size == 1) {
        return reduce_contiguous(data, dim_size);
    } else {
        return reduce_strided(data, dim_size, inner_size);
    }
}

// Main CUDA kernel: each thread computes one output element by reducing over 'dim_size'
// The input tensor layout is assumed to be [outer, dim_size, inner] where the 'dim_size'
// dimension is the one being reduced.
template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) 
{
    int tid = get_global_index();
    int total = outer_size * inner_size;
    if (tid >= total) return;
    
    // Map thread id to outer and inner indices
    int outer_idx = tid / inner_size;
    int inner_idx = tid % inner_size;
    int64_t offset = outer_idx * (dim_size * inner_size) + inner_idx;
    
    // Use unified reduction function
    scalar_t sum = reduce_sum(input + offset, dim_size, inner_size);
    
    output[tid] = sum / static_cast<scalar_t>(dim_size);
}

// Host function to launch the CUDA kernel
// It computes the outer and inner sizes based on the reduction dimension
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }
    
    // Remove the reduced dimension for the output tensor
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Mean reduction (CUDA)");
}
