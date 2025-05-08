#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void max_reduce_kernel_shared_memory(
    const scalar_t* input,
    scalar_t* output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    extern __shared__ scalar_t shared_max[];
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = outer_size * inner_size;
    
    if (idx >= total_elements) return;
    
    const int outer_idx = idx / inner_size;
    const int inner_idx = idx % inner_size;
    
    // Calculate starting position for this thread
    const int64_t start_idx = outer_idx * dim_size * inner_size + inner_idx;
    
    // Initialize with first element
    scalar_t thread_max = input[start_idx];
    
    // Reduce along dimension
    for (int i = 1; i < dim_size; i++) {
        const scalar_t val = input[start_idx + i * inner_size];
        thread_max = max(thread_max, val);
    }
    
    // Use shared memory to reduce within block
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    scalar_t warp_max = thread_max;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        warp_max = max(warp_max, __shfl_down_sync(0xffffffff, warp_max, offset));
    }

    if (lane == 0) {
        shared_max[warp_id] = warp_max;
    }

    __syncthreads();

    if (threadIdx.x < blockDim.x / warpSize) {
        scalar_t block_max = shared_max[threadIdx.x];
        for (int offset = blockDim.x / (2 * warpSize); offset > 0; offset /= 2) {
            block_max = max(block_max, __shfl_down_sync(0xffffffff, block_max, offset));
        }

        if (warp_id == 0 && lane == 0) {
            // Use atomic operation in global memory
            atomicMax(&output[idx], block_max);
        }
    }
}

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    // Handle negative dimension
    if (dim < 0) dim += input.dim();
    
    // Calculate sizes
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }
    
    const int64_t dim_size = input.size(dim);
    
    // Create output tensor
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    const int shared_mem_size = threads / warpSize * sizeof(scalar_t);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        max_reduce_kernel_shared_memory<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward (CUDA)");
}