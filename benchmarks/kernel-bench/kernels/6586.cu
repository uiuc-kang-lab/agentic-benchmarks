#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int BLOCK_SIZE>
__global__ void adaptive_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t num_outputs,
    const bool use_vectorized_load
) {
    for (int out_idx = blockIdx.x; out_idx < num_outputs; out_idx += gridDim.x) {
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;
        int64_t base = outer_idx * dim_size * inner_size + inner_idx;

        int tid = threadIdx.x;
        bool valid = false;
        scalar_t thread_max;

        if (use_vectorized_load && sizeof(scalar_t) == 4) {
            float2* input2 = reinterpret_cast<float2*>(const_cast<scalar_t*>(input));
            float2 val2;
            for (int j = tid; j < dim_size/2; j += BLOCK_SIZE) {
                val2 = input2[base/2 + j * inner_size];
                if (!valid) {
                    thread_max = max(val2.x, val2.y);
                    valid = true;
                } else {
                    thread_max = max(thread_max, max(val2.x, val2.y));
                }
            }
            if (tid == 0 && (dim_size & 1)) {
                scalar_t last_val = input[base + (dim_size-1) * inner_size];
                thread_max = valid ? max(thread_max, last_val) : last_val;
            }
        } else {
            for (int j = tid; j < dim_size; j += BLOCK_SIZE) {
                scalar_t val = input[base + j * inner_size];
                if (!valid) {
                    thread_max = val;
                    valid = true;
                } else {
                    thread_max = max(thread_max, val);
                }
            }
        }

        __shared__ scalar_t shmem[BLOCK_SIZE];
        
        shmem[tid] = valid ? thread_max : input[base];
        __syncthreads();

        if (BLOCK_SIZE > 32) {
            if (tid < 32) {
                #pragma unroll
                for (int i = tid + 32; i < BLOCK_SIZE; i += 32) {
                    shmem[tid] = max(shmem[tid], shmem[i]);
                }
            }
            __syncthreads();
        }

        if (tid < 32) {
            volatile scalar_t* smem = shmem;
            if (BLOCK_SIZE >= 64) smem[tid] = max(smem[tid], smem[tid + 32]);
            if (BLOCK_SIZE >= 32) smem[tid] = max(smem[tid], smem[tid + 16]);
            if (BLOCK_SIZE >= 16) smem[tid] = max(smem[tid], smem[tid + 8]);
            if (BLOCK_SIZE >= 8)  smem[tid] = max(smem[tid], smem[tid + 4]);
            if (BLOCK_SIZE >= 4)  smem[tid] = max(smem[tid], smem[tid + 2]);
            if (BLOCK_SIZE >= 2)  smem[tid] = max(smem[tid], smem[tid + 1]);
        }

        if (tid == 0) {
            output[out_idx] = shmem[0];
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

    int block_size = 256;
    if (dim_size <= 64) block_size = 64;
    else if (dim_size <= 128) block_size = 128;
    else if (dim_size <= 256) block_size = 256;
    else block_size = 512;

    int grid = std::min(num_outputs, static_cast<int64_t>(1024));
    bool use_vectorized = (dim_size >= 128) && (inner_size % 2 == 0);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "adaptive_max_reduce_forward", ([&] {
        switch (block_size) {
            case 64:
                adaptive_max_reduce_kernel<scalar_t, 64><<<grid, block_size>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs, use_vectorized);
                break;
            case 128:
                adaptive_max_reduce_kernel<scalar_t, 128><<<grid, block_size>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs, use_vectorized);
                break;
            case 256:
                adaptive_max_reduce_kernel<scalar_t, 256><<<grid, block_size>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs, use_vectorized);
                break;
            case 512:
                adaptive_max_reduce_kernel<scalar_t, 512><<<grid, block_size>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs, use_vectorized);
                break;
        }
    }));

    return output;
}