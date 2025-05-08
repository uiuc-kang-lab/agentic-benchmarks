#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void simple_max_reduce_kernel(
    const scalar_t* input,
    scalar_t* output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = outer_size * inner_size;
    
    if (idx >= total_elements) return;
    
    const int outer_idx = idx / inner_size;
    const int inner_idx = idx % inner_size;
    const int64_t start_idx = outer_idx * dim_size * inner_size + inner_idx;
    
    scalar_t max_val = input[start_idx];
    for (int i = 1; i < dim_size; i++) {
        max_val = max(max_val, input[start_idx + i * inner_size]);
    }
    output[idx] = max_val;
}

template <typename scalar_t>
__global__ void parallel_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t num_outputs
) {
    for (int out_idx = blockIdx.x; out_idx < num_outputs; out_idx += gridDim.x) {
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;
        int64_t base = outer_idx * dim_size * inner_size + inner_idx;

        int tid = threadIdx.x;
        int block_size = blockDim.x;
        
        scalar_t thread_max = input[base];
        for (int j = tid + 1; j < dim_size; j += block_size) {
            thread_max = max(thread_max, input[base + j * inner_size]);
        }
        
        extern __shared__ char sdata[];
        scalar_t* shmax = reinterpret_cast<scalar_t*>(sdata);
        shmax[tid] = thread_max;
        __syncthreads();

        for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shmax[tid] = max(shmax[tid], shmax[tid + s]);
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[out_idx] = shmax[0];
        }
    }
}

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= input.size(i);
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) inner_size *= input.size(i);

    const int64_t dim_size = input.size(dim);
    const int64_t num_outputs = outer_size * inner_size;
    
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    const bool use_parallel = (dim_size >= 512 || num_outputs <= 1024);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "adaptive_max_reduce_forward", ([&] {
        if (use_parallel) {
            int threads = (dim_size < 256 ? dim_size : 256);
            int blocks = (num_outputs < 1024 ? num_outputs : 1024);
            size_t shm_size = threads * sizeof(scalar_t);
            
            parallel_max_reduce_kernel<scalar_t><<<blocks, threads, shm_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size,
                inner_size,
                num_outputs
            );
        } else {
            const int threads = 256;
            const int blocks = (num_outputs + threads - 1) / threads;
            
            simple_max_reduce_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer_size,
                dim_size,
                inner_size
            );
        }
    }));
    
    return output;
}