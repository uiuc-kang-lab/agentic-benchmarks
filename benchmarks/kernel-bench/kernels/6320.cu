#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sum_reduce_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t numel,
    int64_t reduce_size,
    int64_t outer_size,
    int64_t inner_size) {
    
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int block_offset = blockIdx.x * blockDim.x;
    
    // Each thread handles multiple elements with warp-aligned stride
    const int elements_per_thread = (outer_size * inner_size + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
    
    for (int i = 0; i < elements_per_thread; i++) {
        const int global_idx = block_offset + tid + i * gridDim.x * blockDim.x;
        if (global_idx >= outer_size * inner_size) continue;
        
        const int outer_idx = global_idx / inner_size;
        const int inner_idx = global_idx % inner_size;
        
        scalar_t thread_sum = 0;
        const int64_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;
        
        // Compute local sum
        #pragma unroll 4
        for (int j = 0; j < reduce_size; j++) {
            thread_sum += input[base_idx + j * inner_size];
        }
        
        // Store in shared memory
        shared_data[tid] = thread_sum;
        __syncthreads();
        
        // Warp-level reduction
        if (lane_id == 0) {
            scalar_t warp_sum = shared_data[warp_id * 32];
            #pragma unroll
            for (int j = 1; j < 32; j++) {
                warp_sum += shared_data[warp_id * 32 + j];
            }
            output[global_idx] = warp_sum;
        }
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
    
    const int threads = 256;  // 8 warps per block
    const int blocks = min(256, (int)((outer_size * inner_size + threads - 1) / threads));
    const int shared_mem_size = threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        sum_reduce_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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