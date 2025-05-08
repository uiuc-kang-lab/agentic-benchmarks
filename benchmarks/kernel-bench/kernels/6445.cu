#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {

    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int global_idx = bid * blockDim.x + tid;
    
    if (global_idx >= outer_size * inner_size) return;
    
    const int outer_idx = global_idx / inner_size;
    const int inner_idx = global_idx % inner_size;
    const int input_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    // Initialize accumulator
    scalar_t sum = 0;
    
    // Process dim_size elements in chunks of blockDim.x
    for (int chunk_start = 0; chunk_start < dim_size; chunk_start += blockDim.x) {
        // Load data into shared memory
        scalar_t local_val = 0;
        int chunk_idx = chunk_start + tid;
        
        if (chunk_idx < dim_size) {
            if (inner_size == 1) {
                // Contiguous case: try vectorized loads
                if constexpr (sizeof(scalar_t) == 4 && ((uintptr_t)(input + input_offset + chunk_idx) & 0xF) == 0) {
                    if (chunk_idx + 3 < dim_size) {
                        float4 val = __ldg(reinterpret_cast<const float4*>(input + input_offset + chunk_idx));
                        local_val = val.x + val.y + val.z + val.w;
                    } else {
                        local_val = __ldg(input + input_offset + chunk_idx);
                    }
                } else {
                    local_val = __ldg(input + input_offset + chunk_idx);
                }
            } else {
                // Non-contiguous case
                local_val = __ldg(input + input_offset + chunk_idx * inner_size);
            }
        }
        
        shared_data[tid] = local_val;
        __syncthreads();
        
        // Reduce within the chunk
        if (tid < min(blockDim.x, dim_size - chunk_start)) {
            sum += shared_data[tid];
        }
        __syncthreads();
    }
    
    // Final reduction and write result
    if (tid == 0) {
        output[global_idx] = sum / static_cast<scalar_t>(dim_size);
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
    auto output = torch::empty(sizes, input.options());
    
    // Choose thread block size based on device properties
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    // Calculate shared memory size
    const int shared_mem_size = threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
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