#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sum_reduce_grid_stride_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t numel,
    int64_t reduce_size,
    int64_t outer_size,
    int64_t inner_size) {
    
    const int64_t total_work = outer_size * inner_size;
    const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;

    for (int64_t idx = thread_idx; idx < total_work; idx += stride) {
        const int64_t outer_idx = idx / inner_size;
        const int64_t inner_idx = idx % inner_size;
        
        scalar_t sum = 0;
        const int64_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;
        
        for (int64_t i = 0; i < reduce_size; ++i) {
            sum += input[base_idx + i * inner_size];
        }
        
        output[idx] = sum;
    }
}

torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
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
    
    const int threads = 256;
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    const int blocks = std::min(static_cast<int>((outer_size * inner_size + threads - 1) / threads), num_sms * 4 * 32);  // 4 waves per SM

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        sum_reduce_grid_stride_kernel<scalar_t><<<blocks, threads>>>(
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