#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t numel,
    int64_t reduce_size,
    int64_t outer_size,
    int64_t inner_size) {
    
    // Align block size to warp size (32)
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    const unsigned int total_elements = outer_size * inner_size;
    
    // Grid-stride loop aligned to warps
    for (unsigned int idx = tid; idx < ((total_elements + 31) / 32) * 32; idx += stride) {
        if (idx < total_elements) {
            const unsigned int outer_idx = idx / inner_size;
            const unsigned int inner_idx = idx % inner_size;
            
            scalar_t sum = 0;
            const int64_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;
            
            // Vectorized reduction loop
            #pragma unroll 4
            for (int i = 0; i < reduce_size; i++) {
                sum += input[base_idx + i * inner_size];
            }
            
            output[outer_idx * inner_size + inner_idx] = sum;
        }
    }
}

torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }
    
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());
    
    // Use multiple of warp size for block dimension
    const int threads = 256; // 8 warps per block
    const int blocks = min(65535, (outer_size * inner_size + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        sum_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel(),
            reduce_size,
            outer_size,
            inner_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA)");
}