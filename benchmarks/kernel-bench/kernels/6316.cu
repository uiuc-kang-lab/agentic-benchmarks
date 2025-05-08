#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void optimized_sum_reduce_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t reduce_size,
    int64_t outer_size,
    int64_t inner_size) {
    
    extern __shared__ scalar_t smem[];
    
    const int64_t output_idx = blockIdx.x;
    const int64_t outer_idx = output_idx / inner_size;
    const int64_t inner_idx = output_idx % inner_size;
    const int64_t base = outer_idx * reduce_size * inner_size + inner_idx;

    scalar_t sum = 0;
    for (int64_t i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        sum += input[base + i * inner_size];
    }

    smem[threadIdx.x] = sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Warp-level reduction
    if (threadIdx.x < 32) {
        scalar_t v = smem[threadIdx.x];
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xFFFFFFFF, v, offset);
        }
        if (threadIdx.x == 0) smem[0] = v;
    }

    if (threadIdx.x == 0) {
        output[output_idx] = smem[0];
    }
}

torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= sizes[i];
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) inner_size *= sizes[i];

    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    const int64_t num_blocks = outer_size * inner_size;
    const int threads = 256;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        optimized_sum_reduce_kernel<scalar_t><<<num_blocks, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            outer_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA)");
}