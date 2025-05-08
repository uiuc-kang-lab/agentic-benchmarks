#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel combines the templated block size approach with adaptive workload distribution
// to efficiently perform max reduction across a specified dimension.

template <typename scalar_t, int BLOCK_SIZE>
__global__ void adaptive_blocksize_max_reduce_kernel(
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
        bool valid = false;
        scalar_t thread_max = 0;
        for (int j = tid; j < dim_size; j += BLOCK_SIZE) {
            scalar_t val = input[base + j * inner_size];
            if (!valid) {
                thread_max = val;
                valid = true;
            } else {
                thread_max = max(thread_max, val);
            }
        }

        extern __shared__ char smem[];
        scalar_t* shmem = reinterpret_cast<scalar_t*>(smem);
        shmem[tid] = valid ? thread_max : input[base];
        __syncthreads();

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

torch::Tensor adaptive_max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
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
    const int64_t num_outputs = outer_size * inner_size;

    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    int block_size = (dim_size <= 32) ? 32 : (dim_size <= 64) ? 64 : (dim_size <= 128) ? 128 : (dim_size <= 256) ? 256 : 512;
    int grid = (num_outputs < 1024) ? num_outputs : 1024;
    size_t shm_size = block_size * input.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "adaptive_max_reduce_forward", ([&] {
        switch (block_size) {
            case 32:
                adaptive_blocksize_max_reduce_kernel<scalar_t, 32><<<grid, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
            case 64:
                adaptive_blocksize_max_reduce_kernel<scalar_t, 64><<<grid, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
            case 128:
                adaptive_blocksize_max_reduce_kernel<scalar_t, 128><<<grid, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
            case 256:
                adaptive_blocksize_max_reduce_kernel<scalar_t, 256><<<grid, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
            case 512:
                adaptive_blocksize_max_reduce_kernel<scalar_t, 512><<<grid, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
            default:
                adaptive_blocksize_max_reduce_kernel<scalar_t, 256><<<grid, block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &adaptive_max_reduce_cuda_forward, "Adaptive block size max reduce forward (CUDA)");
}
