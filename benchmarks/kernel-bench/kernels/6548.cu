#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void hybrid_reduce_mean_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {
    
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
    __shared__ scalar_t shared_data[BLOCK_SIZE];
    
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int output_idx = blockIdx.x;
    
    if (output_idx >= outer_size * inner_size) return;
    
    const int outer_idx = output_idx / inner_size;
    const int inner_idx = output_idx % inner_size;
    const int input_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    // First level reduction: threads cooperatively load and sum values
    scalar_t thread_sum = 0;
    #pragma unroll 4
    for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
        thread_sum += input[input_offset + i * inner_size];
    }
    
    // Store partial sum in shared memory
    shared_data[tid] = thread_sum;
    __syncthreads();
    
    // Second level reduction: warp-level reduction of shared memory
    if (tid < WARP_SIZE) {
        scalar_t warp_sum = 0;
        #pragma unroll
        for (int i = tid; i < BLOCK_SIZE; i += WARP_SIZE) {
            warp_sum += shared_data[i];
        }
        
        // Final warp reduction using shuffle
        warp_sum = warp_reduce(warp_sum);
        
        // First thread writes result
        if (lane == 0) {
            output[output_idx] = warp_sum / static_cast<scalar_t>(dim_size);
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
    
    const int BLOCK_SIZE = 256;
    const int num_blocks = outer_size * inner_size;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hybrid_reduce_mean_cuda", ([&] {
        hybrid_reduce_mean_kernel<scalar_t><<<num_blocks, BLOCK_SIZE>>>(
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
    m.def("forward", &mean_reduce_cuda, "Mean reduction using hybrid approach (CUDA)");
}