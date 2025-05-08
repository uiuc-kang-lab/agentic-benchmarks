#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Modular device function for performing sum reduction over a dimension
template <typename scalar_t>
__device__ inline scalar_t compute_sum(const scalar_t* input, int64_t base_idx, int64_t reduce_size, int64_t inner_size) {
    scalar_t sum = 0;
    #pragma unroll
    for (int64_t i = 0; i < reduce_size; i++) {
        sum += input[base_idx + i * inner_size];
    }
    return sum;
}

// Kernel that utilizes the modular device function
template <typename scalar_t>
__global__ void modular_sum_reduce_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t reduce_size,
    int64_t outer_size,
    int64_t inner_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_size * inner_size) return;
    
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    int64_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;
    
    // Use the device function to compute the sum reduction
    scalar_t sum = compute_sum(input, base_idx, reduce_size, inner_size);
    output[outer_idx * inner_size + inner_idx] = sum;
}

// CUDA wrapper function
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
    
    const int threads = 512;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        modular_sum_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            outer_size,
            inner_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Modular sum reduction forward (CUDA)");
}
