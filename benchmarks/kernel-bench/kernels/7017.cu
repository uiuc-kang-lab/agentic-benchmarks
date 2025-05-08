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
    extern __shared__ scalar_t shared_values[];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_slices = outer_size * inner_size;
    if (idx >= total_slices) return;

    int64_t outer = idx / inner_size;
    int64_t inner = idx % inner_size;
    const scalar_t* slice_start = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;

    scalar_t min_val = slice_start[0];
    int min_index = 0;

    const int tile_size = 32;
    for (int k_base = 1; k_base < K; k_base += tile_size) {
        int k_end = min(k_base + tile_size, K);
        
        // Load tile into shared memory
        for (int k = k_base; k < k_end; ++k) {
            shared_values[threadIdx.x * tile_size + (k - k_base)] = slice_start[k * inner_size];
        }
        __syncthreads();

        // Process tile from shared memory
        for (int k = k_base; k < k_end; ++k) {
            scalar_t val = shared_values[threadIdx.x * tile_size + (k - k_base)];
            if (val < min_val) {
                min_val = val;
                min_index = k;
            }
        }
        __syncthreads();
    }

    output[idx] = min_index;
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
    for (int i = 0; i < dims; i++) if (i != dim) out_sizes.push_back(x.size(i));
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

    int64_t total_slices = outer_size * inner_size;
    int threads = 256;
    int blocks = (total_slices + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        size_t shared_mem = tile_size * threads * sizeof(scalar_t);
        argmin_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<int64_t>(),
            K,
            outer_size,
            inner_size
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmin_cuda_forward, "Argmin forward (CUDA)");
}