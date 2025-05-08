#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to perform sum reduction on a segment
template <typename scalar_t>
__device__ scalar_t sum_reduction_segment(
    const scalar_t* input,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t base_idx) {
    scalar_t sum = 0;
    for (int i = 0; i < reduce_size; i++) {
        sum += input[base_idx + i * inner_size];
    }
    return sum;
}

// Kernel function implementing sum reduction using modular device function
template <typename scalar_t>
__global__ void sum_reduce_modular_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t numel,
    int64_t reduce_size,
    int64_t outer_size,
    int64_t inner_size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;
    const int outer_idx = tid / inner_size;
    const int inner_idx = tid % inner_size;
    const int64_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;

    // Using device function for reduction
    scalar_t sum = sum_reduction_segment<scalar_t>(input, reduce_size, inner_size, base_idx);
    output[outer_idx * inner_size + inner_idx] = sum;
}

torch::Tensor sum_reduce_modular_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();
    
    // Calculate sizes
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
    
    // Prepare output tensor
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_modular_cuda", ([&] {
        sum_reduce_modular_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &sum_reduce_modular_cuda, "Sum reduction with modular function (CUDA)");
}