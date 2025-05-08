#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

template <typename scalar_t>
__global__ void argmin_unrolled_kernel(const scalar_t* __restrict__ x,
                                      int64_t* __restrict__ output,
                                      int K,
                                      int64_t inner_size) {
    int inner = blockIdx.x * blockDim.x + threadIdx.x;
    int outer = blockIdx.y;
    if (inner >= inner_size) return;

    // Compute base pointer for this slice
    const scalar_t* slice_ptr = x + static_cast<int64_t>(outer) * (K * inner_size) + inner;
    
    // Initialize min value and index
    scalar_t min_val = slice_ptr[0];
    int min_idx = 0;
    
    // Main loop with manual unrolling by 4
    int k = 1;
    #pragma unroll
    for (; k <= K-4; k += 4) {
        scalar_t val1 = slice_ptr[k * inner_size];
        scalar_t val2 = slice_ptr[(k+1) * inner_size];
        scalar_t val3 = slice_ptr[(k+2) * inner_size];
        scalar_t val4 = slice_ptr[(k+3) * inner_size];
        
        if (val1 < min_val) {
            min_val = val1;
            min_idx = k;
        }
        if (val2 < min_val) {
            min_val = val2;
            min_idx = k + 1;
        }
        if (val3 < min_val) {
            min_val = val3;
            min_idx = k + 2;
        }
        if (val4 < min_val) {
            min_val = val4;
            min_idx = k + 3;
        }
    }
    
    // Handle remaining elements
    #pragma unroll
    for (; k < K; k++) {
        scalar_t val = slice_ptr[k * inner_size];
        if (val < min_val) {
            min_val = val;
            min_idx = k;
        }
    }
    
    output[static_cast<int64_t>(outer) * inner_size + inner] = min_idx;
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    int dims = x.dim();
    if (dim < 0) {
        dim += dims;
    }
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
    
    // Use 2D grid for better occupancy
    int threads_x = 256;
    dim3 block(threads_x);
    dim3 grid((inner_size + threads_x - 1) / threads_x, outer_size);
    
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_unrolled_kernel<scalar_t><<<grid, block>>>(x_data, output_data, K, inner_size);
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