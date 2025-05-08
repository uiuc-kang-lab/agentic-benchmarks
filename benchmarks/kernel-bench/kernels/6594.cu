#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel with dynamic block size selection for optimal performance
// This version uses a templated approach to test different block sizes

template <typename scalar_t, int BLOCK_SIZE>
__global__ void dynamic_blocksize_max_reduce_kernel(
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
        scalar_t thread_max = -INFINITY;
        for (int j = tid; j < dim_size; j += BLOCK_SIZE) {
            scalar_t val = input[base + j * inner_size];
            thread_max = max(thread_max, val);
        }

        extern __shared__ char sdata[];
        scalar_t* shmem = reinterpret_cast<scalar_t*>(sdata);

        shmem[tid] = thread_max;
        __syncthreads();

        for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
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

// Forward function with dynamic block size tuning
// The kernel is launched with different block sizes to find the optimal configuration

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
    const int64_t num_outputs = outer_size * inner_size;

    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // Experiment with different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};
    int best_block_size = 256; // Default initial block size
    float min_time = FLT_MAX;

    for (int block_size : block_sizes) {
        int grid = (num_outputs < 1024) ? num_outputs : 1024;
        size_t shm_size = block_size * input.element_size();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamic_max_reduce_forward", ([&] {
            switch (block_size) {
                case 32:
                    dynamic_blocksize_max_reduce_kernel<scalar_t, 32><<<grid, block_size, shm_size>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        dim_size, inner_size, num_outputs);
                    break;
                case 64:
                    dynamic_blocksize_max_reduce_kernel<scalar_t, 64><<<grid, block_size, shm_size>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        dim_size, inner_size, num_outputs);
                    break;
                case 128:
                    dynamic_blocksize_max_reduce_kernel<scalar_t, 128><<<grid, block_size, shm_size>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        dim_size, inner_size, num_outputs);
                    break;
                case 256:
                    dynamic_blocksize_max_reduce_kernel<scalar_t, 256><<<grid, block_size, shm_size>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        dim_size, inner_size, num_outputs);
                    break;
                case 512:
                    dynamic_blocksize_max_reduce_kernel<scalar_t, 512><<<grid, block_size, shm_size>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        dim_size, inner_size, num_outputs);
                    break;
            }
        }));

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        if (milliseconds < min_time) {
            min_time = milliseconds;
            best_block_size = block_size;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Launch the kernel with the best block size
    int grid = (num_outputs < 1024) ? num_outputs : 1024;
    size_t shm_size = best_block_size * input.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamic_max_reduce_forward", ([&] {
        switch (best_block_size) {
            case 32:
                dynamic_blocksize_max_reduce_kernel<scalar_t, 32><<<grid, best_block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
            case 64:
                dynamic_blocksize_max_reduce_kernel<scalar_t, 64><<<grid, best_block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
            case 128:
                dynamic_blocksize_max_reduce_kernel<scalar_t, 128><<<grid, best_block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
            case 256:
                dynamic_blocksize_max_reduce_kernel<scalar_t, 256><<<grid, best_block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
            case 512:
                dynamic_blocksize_max_reduce_kernel<scalar_t, 512><<<grid, best_block_size, shm_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size, inner_size, num_outputs);
                break;
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward, "Dynamic block size max reduce forward (CUDA)");
}
