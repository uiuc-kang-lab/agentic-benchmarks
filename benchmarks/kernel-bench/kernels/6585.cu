#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int BLOCK_SIZE>
__global__ void hybrid_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t num_outputs
) {
    // Process output elements in a grid-stride loop
    for (int out_idx = blockIdx.x; out_idx < num_outputs; out_idx += gridDim.x) {
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;
        int64_t base = outer_idx * dim_size * inner_size + inner_idx;

        int tid = threadIdx.x;
        scalar_t thread_max = __ldg(&input[base]); // Use __ldg for first read

        // Manual unrolling with aligned memory loads for better coalescing
        const int UNROLL_FACTOR = 4;
        int main_iters = dim_size / (BLOCK_SIZE * UNROLL_FACTOR);
        int offset = tid;

        #pragma unroll
        for (int i = 0; i < main_iters; i++) {
            scalar_t vals[UNROLL_FACTOR];
            #pragma unroll
            for (int j = 0; j < UNROLL_FACTOR; j++) {
                vals[j] = __ldg(&input[base + (offset + j * BLOCK_SIZE) * inner_size]);
            }
            
            #pragma unroll
            for (int j = 0; j < UNROLL_FACTOR; j++) {
                thread_max = max(thread_max, vals[j]);
            }
            offset += BLOCK_SIZE * UNROLL_FACTOR;
        }

        // Handle remaining elements
        for (int j = offset; j < dim_size; j += BLOCK_SIZE) {
            thread_max = max(thread_max, __ldg(&input[base + j * inner_size]));
        }

        // Use shared memory for block-level reduction
        extern __shared__ char smem[];
        scalar_t* shmem = reinterpret_cast<scalar_t*>(smem);
        shmem[tid] = thread_max;
        __syncthreads();

        // Tree-based reduction with sequential addressing
        #pragma unroll
        for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shmem[tid] = max(shmem[tid], shmem[tid + s]);
            }
            __syncthreads();
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

    // Dynamic block size selection based on reduction dimension
    int block_size = 256;
    if (dim_size <= 512) block_size = 128;
    else if (dim_size <= 1024) block_size = 256;
    else block_size = 512;

    const int grid_size = std::min(1024, (int)num_outputs);
    const size_t shm_size = block_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "hybrid_max_reduce_forward", ([&] {
        switch(block_size) {
            case 128:
                hybrid_max_reduce_kernel<scalar_t, 128><<<grid_size, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
            case 256:
                hybrid_max_reduce_kernel<scalar_t, 256><<<grid_size, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
            case 512:
                hybrid_max_reduce_kernel<scalar_t, 512><<<grid_size, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
        }
    }));

    return output;
}