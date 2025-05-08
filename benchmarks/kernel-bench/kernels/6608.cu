#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tiled_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t inner_size,
    const int64_t dim_size
) {
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_mem[];
    scalar_t* shared = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int outer_idx = blockIdx.x;
    const int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (inner_idx >= inner_size) return;
    
    const int64_t base_offset = outer_idx * dim_size * inner_size;
    const int tile_size = blockDim.y;
    const int tid = threadIdx.y;
    
    scalar_t max_val = input[base_offset + inner_idx];
    
    // Process reduction dimension in tiles
    for (int tile_start = 1; tile_start < dim_size; tile_start += tile_size) {
        scalar_t tile_max = max_val;
        const int tile_end = min(static_cast<int>(tile_start + tile_size), static_cast<int>(dim_size));
        
        // Load tile elements
        for (int i = tile_start + tid; i < tile_end; i += tile_size) {
            tile_max = max(tile_max, input[base_offset + i * inner_size + inner_idx]);
        }
        
        // Store partial max in shared memory
        shared[threadIdx.x * tile_size + tid] = tile_max;
        __syncthreads();
        
        // Reduce within tile
        for (int s = tile_size/2; s > 0; s >>= 1) {
            if (tid < s) {
                shared[threadIdx.x * tile_size + tid] = max(
                    shared[threadIdx.x * tile_size + tid],
                    shared[threadIdx.x * tile_size + tid + s]
                );
            }
            __syncthreads();
        }
        
        // Update global max
        if (tid == 0) {
            max_val = max(max_val, shared[threadIdx.x * tile_size]);
        }
        __syncthreads();
    }
    
    // Write final result
    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = max_val;
    }
}

torch::Tensor tiled_max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= input.size(i);
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) inner_size *= input.size(i);
    
    const int64_t dim_size = input.size(dim);
    
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());
    
    const int TILE_SIZE = 32;
    const int INNER_THREADS = 256;
    dim3 blocks(outer_size, (inner_size + INNER_THREADS - 1) / INNER_THREADS);
    dim3 threads(INNER_THREADS, TILE_SIZE);
    size_t shared_size = INNER_THREADS * TILE_SIZE * sizeof(typename torch::ScalarTypeToCPPType<torch::kFloat>::type);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "tiled_max_reduce", ([&] {
        tiled_max_reduce_kernel<scalar_t><<<blocks, threads, shared_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            inner_size,
            dim_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &tiled_max_reduce_cuda_forward, "Tiled max reduce forward (CUDA)");
}