#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// Utility function to perform warp-wide minimum reduction
template <typename scalar_t>
__device__ scalar_t warp_reduce_min(scalar_t val, int& min_index, int lane, int* warp_min_index) {
    for (int offset = 16; offset > 0; offset /= 2) {
        scalar_t other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        int other_index = __shfl_down_sync(0xFFFFFFFF, min_index, offset);
        if (other < val) {
            val = other;
            min_index = other_index;
        }
    }
    if (lane == 0) warp_min_index[threadIdx.x / 32] = min_index;
    return val;
}

// Kernel leveraging warp intrinsics
template <typename scalar_t>
__global__ void argmin_warp_kernel(const scalar_t* __restrict__ x,
                                   int64_t* __restrict__ output,
                                   int K,
                                   int64_t outer_size,
                                   int64_t inner_size) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= outer_size * inner_size) return;

    int64_t outer = idx / inner_size;
    int64_t inner = idx % inner_size;
    const scalar_t* slice_ptr = x + outer * K * inner_size + inner;

    scalar_t min_val = slice_ptr[0];
    int min_index = 0;
    for (int k = 1; k < K; ++k) {
        scalar_t val = slice_ptr[k * inner_size];
        if (val < min_val) {
            min_val = val;
            min_index = k;
        }
    }

    int lane = threadIdx.x % 32;
    __shared__ int warp_min_index[8]; // Assuming 256 threads and 8 warps per block
    scalar_t warp_min = warp_reduce_min(min_val, min_index, lane, warp_min_index);

    if (lane == 0) {
        int block_min_index = warp_min_index[0];
        for (int i = 1; i < 8; ++i) {
            if (slice_ptr[warp_min_index[i] * inner_size] < slice_ptr[block_min_index * inner_size]) {
                block_min_index = warp_min_index[i];
            }
        }
        output[idx] = block_min_index;
    }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    int dims = x.dim();
    if (dim < 0) dim += dims;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= x.size(i);
    int K = static_cast<int>(x.size(dim));
    int64_t inner_size = 1;
    for (int i = dim + 1; i < dims; i++) inner_size *= x.size(i);

    std::vector<int64_t> out_sizes;
    for (int i = 0; i < dims; i++) if (i != dim) out_sizes.push_back(x.size(i));
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

    int64_t total_slices = outer_size * inner_size;
    int threads = 256;
    int blocks = (total_slices + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        argmin_warp_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<int64_t>(),
            K,
            outer_size,
            inner_size
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmin_cuda_forward, "Argmin forward (CUDA)");
}