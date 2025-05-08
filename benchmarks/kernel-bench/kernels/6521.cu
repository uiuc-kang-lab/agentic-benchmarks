#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t outer_size,
    int64_t dim_size, 
    int64_t inner_size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= outer_size * inner_size) return;
    
    const int outer_idx = tid / inner_size;
    const int inner_idx = tid % inner_size;
    const int input_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    scalar_t sum = 0;
    for (int i = 0; i < dim_size; i++) {
        sum += input[input_offset + i * inner_size];
    }
    
    output[tid] = sum / dim_size;
}

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }
    
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());
    
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Mean reduction (CUDA)");
}