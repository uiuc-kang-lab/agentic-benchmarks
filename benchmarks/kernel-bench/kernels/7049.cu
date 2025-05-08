#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
__global__ void min_reduce_warp_shared_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {
    
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int warp_size = 32;
    const int warps_per_block = blockDim.x / warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int lane_id = threadIdx.x % warp_size;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    
    if (global_warp_id >= outer * inner) return;
    
    const int outer_idx = global_warp_id / inner;
    const int inner_idx = global_warp_id % inner;
    const int base = outer_idx * (r * inner) + inner_idx;
    
    const int tile_size = 128;
    scalar_t warp_min = std::numeric_limits<scalar_t>::max();
    
    #pragma unroll 1
    for (int tile_start = 0; tile_start < r; tile_start += tile_size) {
        const int tile_end = min(tile_start + tile_size, r);
        
        for (int j = tile_start + lane_id; j < tile_end; j += warp_size) {
            const int shared_idx = (j - tile_start) + warp_id * tile_size;
            const int global_idx = base + j * inner;
            shared_data[shared_idx] = input[global_idx];
        }
        __syncthreads();
        
        for (int j = 0; j < (tile_end - tile_start); j++) {
            const int shared_idx = j + warp_id * tile_size;
            scalar_t val = shared_data[shared_idx];
            warp_min = min(warp_min, val);
        }
        __syncthreads();
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        scalar_t other = __shfl_down_sync(0xffffffff, warp_min, offset);
        warp_min = min(warp_min, other);
    }
    
    if (lane_id == 0) {
        output[global_warp_id] = warp_min;
    }
}

torch::Tensor forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    int ndim = input.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    int outer = 1;
    for (int i = 0; i < dim; i++) {
        outer *= input.size(i);
    }
    int r = input.size(dim);
    int inner = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner *= input.size(i);
    }

    std::vector<int64_t> output_shape;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            output_shape.push_back(input.size(i));
        }
    }
    
    auto output = torch::empty(output_shape, input.options());

    const int threads_per_block = 128;
    const int warps_per_block = threads_per_block / 32;
    const int num_warps = outer * inner;
    const int num_blocks = (num_warps + warps_per_block - 1) / warps_per_block;
    
    const int shared_mem_size = 128 * warps_per_block * sizeof(float);

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_warp_shared_cuda", ([&] {
        min_reduce_warp_shared_kernel<scalar_t><<<num_blocks, threads_per_block, shared_mem_size,
            c10::cuda::getCurrentCUDAStream().stream()>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer,
            r,
            inner);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Min reduction over a specified dimension using warp-level primitives with shared memory (CUDA)");
}