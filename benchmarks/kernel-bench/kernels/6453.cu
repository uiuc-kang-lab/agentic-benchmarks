#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

template <typename scalar_t>
__inline__ __device__
scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {
    
    extern __shared__ char shared_memory[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_memory);
    
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lane = tid % 32;
    const int warps_per_block = blockDim.x / 32;
    
    const int64_t global_idx = blockIdx.x * blockDim.x + tid;
    const int64_t total_threads = outer_size * inner_size;
    
    if (global_idx < total_threads) {
        const int64_t outer_idx = global_idx / inner_size;
        const int64_t inner_idx = global_idx % inner_size;
        const int64_t input_offset = outer_idx * dim_size * inner_size + inner_idx;
        
        scalar_t thread_sum = 0;
        
        const int thread_stride = (dim_size + blockDim.x - 1) / blockDim.x;
        const int thread_start = tid * thread_stride;
        const int thread_end = min(static_cast<int>(thread_start + thread_stride), static_cast<int>(dim_size));
        
        if (inner_size == 1 && (reinterpret_cast<uintptr_t>(input + input_offset) & 0xF) == 0) {
            if constexpr (sizeof(scalar_t) == 4) {
                const float4* input4 = reinterpret_cast<const float4*>(input + input_offset + thread_start);
                for (int i = thread_start; i < thread_end - 3; i += 4) {
                    float4 val = __ldg(input4++);
                    thread_sum += val.x + val.y + val.z + val.w;
                }
                for (int i = (thread_end & ~3); i < thread_end; i++) {
                    thread_sum += __ldg(input + input_offset + i);
                }
            } else {
                for (int i = thread_start; i < thread_end; i++) {
                    thread_sum += __ldg(input + input_offset + i * inner_size);
                }
            }
        } else {
            for (int i = thread_start; i < thread_end; i++) {
                thread_sum += __ldg(input + input_offset + i * inner_size);
            }
        }
        
        thread_sum = warp_reduce_sum(thread_sum);
        
        if (lane == 0) {
            shared_data[wid] = thread_sum;
        }
        __syncthreads();
        
        if (wid == 0 && lane < warps_per_block) {
            thread_sum = shared_data[lane];
            thread_sum = warp_reduce_sum(thread_sum);
            
            if (lane == 0) {
                output[global_idx] = thread_sum / static_cast<scalar_t>(dim_size);
            }
        }
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
    
    const int threads_per_block = 256;
    const int blocks = (outer_size * inner_size + threads_per_block - 1) / threads_per_block;
    const int warps_per_block = threads_per_block / 32;
    const int shared_memory_size = warps_per_block * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads_per_block, shared_memory_size>>>(
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
    m.def("forward", &mean_reduce_cuda, "Warp-optimized mean reduction (CUDA)");
}