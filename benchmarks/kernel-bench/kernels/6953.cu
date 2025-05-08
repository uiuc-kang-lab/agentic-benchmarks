#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

template <typename scalar_t>
__global__ void argmin_coalesced_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size) {
    
    const int tid = threadIdx.x;
    const int outer = blockIdx.x;
    const int inner_chunk = blockIdx.y;
    
    const int block_size = blockDim.x;
    const int64_t inner_start = inner_chunk * block_size;
    const int64_t inner = inner_start + tid;
    
    if (outer >= outer_size || inner >= inner_size) return;
    
    scalar_t min_val = std::numeric_limits<scalar_t>::max();
    int min_index = 0;
    
    const scalar_t* outer_ptr = x + outer * K * inner_size;
    
    for (int k = 0; k < K; ++k) {
        scalar_t val = __ldg(outer_ptr + k * inner_size + inner);
        if (val < min_val) {
            min_val = val;
            min_index = k;
        }
    }
    
    output[outer * inner_size + inner] = min_index;
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
    
    const int block_size = 256;
    const int inner_chunks = (inner_size + block_size - 1) / block_size;
    dim3 grid(outer_size, inner_chunks);
    
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_coalesced_kernel<scalar_t><<<grid, block_size>>>(x_data, output_data, K, outer_size, inner_size);
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
