#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

template <typename scalar_t, int BLOCK_SIZE>
__global__ void optimized_min_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {

    int idx = blockIdx.x;
    if (idx >= outer * inner) return;

    int outer_idx = idx / inner;
    int inner_idx = idx % inner;
    const scalar_t* in_ptr = input + outer_idx * (r * inner) + inner_idx;

    scalar_t local_min = std::numeric_limits<scalar_t>::max();
    
    #pragma unroll 4
    for (int j = threadIdx.x; j < r; j += BLOCK_SIZE) {
        // Use __ldg for read-only access with caching and 128-bit alignment
        scalar_t val = __ldg(in_ptr + j * inner);
        local_min = min(local_min, val);
    }

    __shared__ scalar_t sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = local_min;
    __syncthreads();

    if (BLOCK_SIZE >= 512) { if (threadIdx.x < 256) { sdata[threadIdx.x] = min(sdata[threadIdx.x], sdata[threadIdx.x + 256]); } __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (threadIdx.x < 128) { sdata[threadIdx.x] = min(sdata[threadIdx.x], sdata[threadIdx.x + 128]); } __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (threadIdx.x < 64) { sdata[threadIdx.x] = min(sdata[threadIdx.x], sdata[threadIdx.x + 64]); } __syncthreads(); }
    
    if (threadIdx.x < 32) {
        volatile scalar_t* vsdata = sdata;
        if (BLOCK_SIZE >= 64) vsdata[threadIdx.x] = min(vsdata[threadIdx.x], vsdata[threadIdx.x + 32]);
        vsdata[threadIdx.x] = min(vsdata[threadIdx.x], vsdata[threadIdx.x + 16]);
        vsdata[threadIdx.x] = min(vsdata[threadIdx.x], vsdata[threadIdx.x + 8]);
        vsdata[threadIdx.x] = min(vsdata[threadIdx.x], vsdata[threadIdx.x + 4]);
        vsdata[threadIdx.x] = min(vsdata[threadIdx.x], vsdata[threadIdx.x + 2]);
        vsdata[threadIdx.x] = min(vsdata[threadIdx.x], vsdata[threadIdx.x + 1]);
    }

    if (threadIdx.x == 0) {
        output[idx] = sdata[0];
    }
}

torch::Tensor forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    input = input.contiguous();

    int ndim = input.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    int outer = 1;
    for (int i = 0; i < dim; i++) outer *= input.size(i);
    int r = input.size(dim);
    int inner = 1;
    for (int i = dim + 1; i < ndim; i++) inner *= input.size(i);

    auto output = torch::empty((dim == 0) ? input.sizes().slice(1) : input.sizes().slice(0, dim),
                              input.options());
    int total = outer * inner;

    int block_size = r <= 256 ? (r <= 128 ? (r <= 64 ? 32 : 64) : 128) : 256;

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "optimized_min_reduce_cuda", ([&] {
        switch(block_size) {
            case 32:
                optimized_min_reduction_kernel<scalar_t, 32><<<total, 32, 0>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), outer, r, inner);
                break;
            case 64:
                optimized_min_reduction_kernel<scalar_t, 64><<<total, 64, 0>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), outer, r, inner);
                break;
            case 128:
                optimized_min_reduction_kernel<scalar_t, 128><<<total, 128, 0>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), outer, r, inner);
                break;
            default:
                optimized_min_reduction_kernel<scalar_t, 256><<<total, 256, 0>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), outer, r, inner);
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized vector load min reduction (CUDA)");
}