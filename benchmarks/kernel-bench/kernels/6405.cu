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

// Kernel function implementing sum reduction using modular device function with shared memory optimization
template <typename scalar_t>
__global__ void sum_reduce_shared_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t reduce_size,
    int64_t inner_size) {
    extern __shared__ scalar_t shared_data[];
    const int tid = threadIdx.x;
    const int seg_id = blockIdx.x;
    const int outer_idx = seg_id / inner_size;
    const int inner_idx = seg_id % inner_size;
    const int64_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;

    scalar_t sum = 0;
    for (int i = tid; i < reduce_size; i += blockDim.x) {
        sum += input[base_idx + i * inner_size];
    }
    shared_data[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[seg_id] = shared_data[0];
    }
}

torch::Tensor sum_reduce_shared_cuda(torch::Tensor input, int64_t dim) {
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
    const int shared_memory_size = threads * sizeof(scalar_t);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_shared_cuda", ([&] {
        sum_reduce_shared_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
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
    m.def("forward", &sum_reduce_shared_cuda, "Sum reduction with shared memory optimization (CUDA)");
}