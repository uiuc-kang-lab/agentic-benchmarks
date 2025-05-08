#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void mean_reduce_kernel_warp_optimized(
    const scalar_t* input,
    scalar_t* output,
    int64_t outer_size,
    int64_t dim_size, 
    int64_t inner_size) {
    
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
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        sum += input[input_offset + i * inner_size];
    }

    // Use shared memory to accumulate results within a block
    __shared__ scalar_t shared_sum[256];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (threadIdx.x == 0) {
        output[tid] = shared_sum[0] / dim_size;
    }
}

torch::Tensor mean_reduce_cuda_warp_optimized(torch::Tensor input, int64_t dim) {
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
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "mean_reduce_cuda_warp_optimized", ([&] {
        mean_reduce_kernel_warp_optimized<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &mean_reduce_cuda_warp_optimized, "Mean reduction warp optimized (CUDA)");
}