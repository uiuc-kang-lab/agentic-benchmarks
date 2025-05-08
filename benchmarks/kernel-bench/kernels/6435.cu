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
    
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = bid * blockDim.x + tid;
    
    if (gid >= outer_size * inner_size) return;
    
    const int outer_idx = gid / inner_size;
    const int inner_idx = gid % inner_size;
    const int input_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    // Initialize shared memory
    shared_data[tid] = 0;
    __syncthreads();
    
    // Each thread accumulates its portion
    scalar_t local_sum = 0;
    for (int i = 0; i < dim_size; i++) {
        local_sum += input[input_offset + i * inner_size];
    }
    
    // Store in shared memory
    shared_data[tid] = local_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Only thread 0 writes the result
    if (tid == 0) {
        atomicAdd(&output[gid], shared_data[0] / dim_size);
    }
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
    auto output = torch::zeros(sizes, input.options());
    
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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