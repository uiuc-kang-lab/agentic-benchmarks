#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
__device__ __inline__ scalar_t warp_reduce_min(scalar_t val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        scalar_t tmp = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = tmp < val ? tmp : val;
    }
    return val;
}

template <typename scalar_t>
__global__ void warp_min_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;

    const int outer_idx = idx / inner;
    const int inner_idx = idx % inner;
    const scalar_t* group_ptr = input + outer_idx * (r * inner) + inner_idx;

    scalar_t min_val = std::numeric_limits<scalar_t>::max();
    for (int j = threadIdx.x % 32; j < r; j += 32) {
        scalar_t val = group_ptr[j * inner];
        if (val < min_val) min_val = val;
    }

    min_val = warp_reduce_min(min_val);

    if (threadIdx.x % 32 == 0) {
        atomicMinBlock(output + idx / blockDim.x * inner + inner_idx, min_val);
    }
}

template <typename scalar_t>
__device__ __inline__ void atomicMinBlock(scalar_t* address, scalar_t val) {
    // Forward declare atomicMin for different types
    scalar_t old = *address;
    scalar_t assumed;
    do {
        assumed = old;
        old = atomicCAS(address, assumed, min(val, assumed));
    } while (assumed != old);
}

torch::Tensor forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    input = input.contiguous();

    int ndim = input.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    int outer = 1;
    for (int i = 0; i < dim; ++i)
        outer *= input.size(i);
    int r = input.size(dim);
    int inner = 1;
    for (int i = dim + 1; i < ndim; ++i)
        inner *= input.size(i);

    auto output = torch::empty({outer, inner}, input.options());

    const int threads_per_block = 256;
    const int blocks = (outer * inner + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "warp_min_reduce", ([&] {
        warp_min_reduce_kernel<scalar_t><<<blocks, threads_per_block, 0, 
            c10::cuda::getCurrentCUDAStream().stream()>>>(input.data_ptr<scalar_t>(),
                                                          output.data_ptr<scalar_t>(),
                                                          outer, r, inner);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-level min reduction (CUDA)");
}
