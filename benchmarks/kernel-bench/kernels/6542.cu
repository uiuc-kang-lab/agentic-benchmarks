#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using atomic operations to handle race conditions especially
// when summing across reduction dimensions using a parallel reduction approach.

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void atomic_mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {

    extern __shared__ float sdata[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= outer_size * inner_size) return;

    int outer_idx = tid / inner_size;
    int inner_idx = tid % inner_size;
    int input_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        local_sum += input[input_offset + i * inner_size];
    }

    local_sum = warp_reduce_sum(local_sum);

    if (threadIdx.x % warpSize == 0) {
        atomicAdd(&sdata[threadIdx.x], local_sum);
    }
    __syncthreads();

    // Assuming a warp per block handles each output
    if (threadIdx.x == 0) {
        atomicAdd(&output[tid], sdata[threadIdx.y] / float(dim_size));
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
    
    // Configure blocks and grids
    dim3 block(32, 1);
    int grid_x = (outer_size * inner_size + block.x - 1) / block.x;
    dim3 grid(grid_x);

    size_t shared_mem_size = block.x * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "atomic_mean_reduce_cuda", ([&] {
        atomic_mean_reduce_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
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
    m.def("forward", &mean_reduce_cuda, "Mean reduction with atomic optimization (CUDA)");
}
