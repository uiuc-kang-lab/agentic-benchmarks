#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to perform block-level reduction using shared memory
template <typename scalar_t>
__device__ __forceinline__ scalar_t block_reduce_sum(scalar_t* sdata, int tid, int blockDim) {
    // Reduce within the block using loop unrolling
    for (int stride = blockDim / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid < 32) {
        volatile scalar_t* vsmem = sdata;
        if (blockDim >= 64) vsmem[tid] += vsmem[tid + 32];
        if (blockDim >= 32) vsmem[tid] += vsmem[tid + 16];
        if (blockDim >= 16) vsmem[tid] += vsmem[tid + 8];
        if (blockDim >= 8)  vsmem[tid] += vsmem[tid + 4];
        if (blockDim >= 4)  vsmem[tid] += vsmem[tid + 2];
        if (blockDim >= 2)  vsmem[tid] += vsmem[tid + 1];
    }
    return sdata[0];
}

// CUDA kernel for mean reduction over a specified dimension
template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {

    extern __shared__ char shared_mem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(shared_mem);

    const int tid = threadIdx.x;
    const int out_idx = blockIdx.x;

    // Check if the current block corresponds to a valid output element
    if (out_idx >= outer_size * inner_size)
        return;

    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;
    int base_idx = outer_idx * dim_size * inner_size + inner_idx;

    // Each thread computes a partial sum over elements in the reduction dimension
    scalar_t sum = 0;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        sum += input[base_idx + i * inner_size];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Use the modular block reduction function to sum the partial results
    scalar_t block_sum = block_reduce_sum(sdata, tid, blockDim.x);
    if (tid == 0) {
        output[out_idx] = block_sum / static_cast<scalar_t>(dim_size);
    }
}

// Host function to prepare and launch the CUDA kernel
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }

    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Remove the reduced dimension from the output shape
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    const int threads = 256;
    const int blocks = outer_size * inner_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        const int shared_mem_size = threads * sizeof(scalar_t);
        mean_reduce_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &mean_reduce_cuda, "Mean reduction (CUDA)");
}
