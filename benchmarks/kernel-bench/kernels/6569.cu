#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void optimized_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = outer_size * inner_size;
    if (idx >= total_elements) return;

    const int outer_idx = idx / inner_size;
    const int inner_idx = idx % inner_size;
    const int64_t base = outer_idx * dim_size * inner_size + inner_idx;

    scalar_t max_val = input[base];
    const int count = dim_size - 1;
    int i = 1;

    #pragma unroll
    for (; i <= count - 4; i += 4) {
        scalar_t vals[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            vals[j] = input[base + (i + j) * inner_size];
        }
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            max_val = max(max_val, vals[j]);
        }
    }

    for (; i <= count; i++) {
        max_val = max(max_val, input[base + i * inner_size]);
    }

    shared_data[threadIdx.x] = max_val;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] = max(shared_data[threadIdx.x], 
                                         shared_data[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }
    
    const int64_t dim_size = input.size(dim);
    
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());
    
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        optimized_max_reduce_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));
    
    return output;
}