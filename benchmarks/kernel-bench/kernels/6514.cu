#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ float float_sum(float x, float y) { return x + y; }

__device__ double double_sum(double x, double y) { return x + y; }

template <typename scalar_t, typename Function>
__global__ void optimized_mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* output,
    int64_t outer_size,
    int64_t dim_size, 
    int64_t inner_size,
    Function sum_func) {
    
    // Calculate global thread index
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread handles one output element
    if (tid >= outer_size * inner_size) return;
    
    // Calculate input/output positions
    const int outer_idx = tid / inner_size;
    const int inner_idx = tid % inner_size;
    const int input_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    // Compute mean by summing over reduction dimension
    scalar_t sum = 0;
    for (int i = 0; i < dim_size; i++) {
        sum = sum_func(sum, __ldg(input + input_offset + i * inner_size));
    }
    
    output[tid] = sum / dim_size;
}

torch::Tensor optimized_mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();
    
    // Calculate sizes
    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];
    
    // Calculate outer and inner sizes
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }
    
    // Create output tensor
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "optimized_mean_reduce_cuda", ([&] {
        auto sum_func = (std::is_same<scalar_t, float>::value) ? float_sum : double_sum;
        optimized_mean_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size,
            sum_func
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_mean_reduce_cuda, "Mean reduction with optimized memory access (CUDA)");
}