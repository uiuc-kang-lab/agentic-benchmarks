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
    
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int output_idx = bid;
    
    if (output_idx >= outer_size * inner_size) return;
    
    const unsigned int outer_idx = output_idx / inner_size;
    const unsigned int inner_idx = output_idx % inner_size;
    const unsigned int input_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    // Load and sum elements using vectorized loads when possible
    scalar_t thread_sum = 0;
    
    // Use padding to avoid bank conflicts in shared memory
    const unsigned int padded_tid = tid + (tid / 32) * 8;
    
    if (sizeof(scalar_t) == sizeof(float) && dim_size >= 4) {
        const float4* input4 = reinterpret_cast<const float4*>(&input[input_offset]);
        float4 sum4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        #pragma unroll 2
        for (unsigned int i = tid; i < dim_size/4; i += blockDim.x) {
            float4 val = __ldg(&input4[i * inner_size/4]);
            sum4.x += val.x;
            sum4.y += val.y;
            sum4.z += val.z;
            sum4.w += val.w;
        }
        thread_sum = sum4.x + sum4.y + sum4.z + sum4.w;
        
        // Handle remaining elements
        #pragma unroll
        for (unsigned int i = (dim_size/4)*4 + tid; i < dim_size; i += blockDim.x) {
            thread_sum += __ldg(&input[input_offset + i * inner_size]);
        }
    } else {
        #pragma unroll 8
        for (unsigned int i = tid; i < dim_size; i += blockDim.x) {
            thread_sum += __ldg(&input[input_offset + i * inner_size]);
        }
    }
    
    // Store the sum in shared memory with padding to avoid bank conflicts
    shared_data[padded_tid] = thread_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory with template unrolling
    if (blockDim.x >= 512) { if (tid < 256) { shared_data[tid] += shared_data[tid + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { shared_data[tid] += shared_data[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid < 64) { shared_data[tid] += shared_data[tid + 64]; } __syncthreads(); }
    
    // Last warp reduction (no sync needed)
    if (tid < 32) {
        volatile scalar_t* smem = shared_data;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8) smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4) smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2) smem[tid] += smem[tid + 1];
    }
    
    // Write result
    if (tid == 0) {
        output[output_idx] = shared_data[0] / static_cast<scalar_t>(dim_size);
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
    
    const int threads = 256;
    const int blocks = outer_size * inner_size;
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
