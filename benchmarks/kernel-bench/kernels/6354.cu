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
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int warp_size = 32;
    
    // Grid-stride loop to handle all elements
    for (int idx = tid; idx < outer_size * inner_size; idx += stride) {
        const int outer_idx = idx / inner_size;
        const int inner_idx = idx % inner_size;
        
        scalar_t sum = 0;
        const int64_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;
        
        // Align reduction loop to warp size for better coherence
        int i = 0;
        for (; i + warp_size <= reduce_size; i += warp_size) {
            #pragma unroll
            for (int j = 0; j < warp_size; j++) {
                sum += input[base_idx + (i + j) * inner_size];
            }
        }
        
        // Handle remaining elements
        for (; i < reduce_size; i++) {
            sum += input[base_idx + i * inner_size];
        }
        
        output[outer_idx * inner_size + inner_idx] = sum;
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
    
    const int threads = 256;
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