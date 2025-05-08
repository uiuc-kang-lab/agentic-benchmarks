#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// Constant memory for frequently accessed parameters
__constant__ int d_K;
__constant__ int64_t d_outer_size;
__constant__ int64_t d_inner_size;

template <typename scalar_t>
__device__ __forceinline__ const scalar_t* get_slice_ptr(const scalar_t* x,
                                                        int64_t outer,
                                                        int64_t inner) {
    return x + outer * (static_cast<int64_t>(d_K) * d_inner_size) + inner;
}

template <typename scalar_t>
__device__ __forceinline__ int compute_argmin(const scalar_t* slice_ptr) {
    scalar_t min_val = slice_ptr[0];
    int min_idx = 0;
    
    #pragma unroll 4
    for (int k = 1; k < d_K; ++k) {
        scalar_t val = slice_ptr[k * d_inner_size];
        if (val < min_val) {
            min_val = val;
            min_idx = k;
        }
    }
    return min_idx;
}

template <typename scalar_t>
__global__ void argmin_kernel(const scalar_t* __restrict__ x,
                             int64_t* __restrict__ output) {
    // Use a grid-stride loop to improve occupancy and utilization
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < d_outer_size * d_inner_size;
         idx += blockDim.x * gridDim.x) {

        int64_t outer = idx / d_inner_size;
        int64_t inner = idx % d_inner_size;

        const scalar_t* slice_ptr = get_slice_ptr(x, outer, inner);
        output[idx] = compute_argmin(slice_ptr);
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

    // Copy parameters to constant memory
    cudaMemcpyToSymbol(d_K, &K, sizeof(int));
    cudaMemcpyToSymbol(d_outer_size, &outer_size, sizeof(int64_t));
    cudaMemcpyToSymbol(d_inner_size, &inner_size, sizeof(int64_t));

    std::vector<int64_t> out_sizes;
    for (int i = 0; i < dims; i++) {
        if (i != dim) out_sizes.push_back(x.size(i));
    }
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

    int64_t total_slices = outer_size * inner_size;
    int threads = 256;
    int blocks = (total_slices + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_kernel<scalar_t><<<blocks, threads>>>(x_data, output_data);
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