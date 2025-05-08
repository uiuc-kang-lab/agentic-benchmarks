#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

template <typename scalar_t>
__global__ void argmin_kernel(const scalar_t* __restrict__ x,
                              int64_t* __restrict__ output,
                              int K,
                              int64_t outer_size,
                              int64_t inner_size) {
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_slices = outer_size * inner_size;
    if (idx >= total_slices) return;

    int64_t outer = idx / inner_size;
    int64_t inner = idx % inner_size;
    const scalar_t* slice_start = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;

    // Load data into shared memory
    for (int k = 0; k < K; k++) {
        shared_data[threadIdx.x * K + k] = slice_start[k * inner_size];
    }
    __syncthreads();

    // Find minimum in shared memory
    scalar_t min_val = shared_data[threadIdx.x * K];
    int min_index = 0;
    
    #pragma unroll
    for (int k = 1; k < K; k++) {
        scalar_t val = shared_data[threadIdx.x * K + k];
        if (val < min_val) {
            min_val = val;
            min_index = k;
        }
    }
    output[idx] = min_index;
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

    int dims = x.dim();
    if (dim < 0) dim += dims;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }
    int K = static_cast<int>(x.size(dim));
    int64_t inner_size = 1;
    for (int i = dim + 1; i < dims; i++) {
        inner_size *= x.size(i);
    }

    std::vector<int64_t> out_sizes;
    for (int i = 0; i < dims; i++) {
        if (i == dim) continue;
        out_sizes.push_back(x.size(i));
    }
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

    int64_t total_slices = outer_size * inner_size;
    int threads = 256;
    int blocks = (total_slices + threads - 1) / threads;
    
    // Shared memory size per block
    size_t shared_mem_size = threads * K * sizeof(typename std::conditional<std::is_same<at::Half, typename std::decay<decltype(x.scalar_type())>::type>::value, half, float>::type);

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(x_data, output_data, K, outer_size, inner_size);
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