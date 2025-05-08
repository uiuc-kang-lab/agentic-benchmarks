#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void optimized_sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t reduce_size,
    const int64_t outer_size,
    const int64_t inner_size) {
    
    const int64_t total_elements = outer_size * inner_size;
    const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;

    for (int64_t idx = thread_idx; idx < total_elements; idx += stride) {
        const int64_t outer_idx = idx / inner_size;
        const int64_t inner_idx = idx % inner_size;
        
        scalar_t sum = 0;
        const int64_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;
        
        #pragma unroll(4)
        for (int i = 0; i < reduce_size; ++i) {
            sum += input[base_idx + i * inner_size];
        }
        
        output[idx] = sum;
    }
}

torch::Tensor optimized_sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    const int64_t reduce_size = sizes[dim];
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= sizes[i];
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); ++i) {
        inner_size *= sizes[i];
    }
    
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());
    
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    
    const int threads = 256;
    const int blocks = std::max(static_cast<int>((outer_size * inner_size + threads - 1) / threads), num_sms * 4);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_sum_reduce", ([&] {
        optimized_sum_reduce_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &optimized_sum_reduce_cuda, "Optimized sum reduction (CUDA)");
}