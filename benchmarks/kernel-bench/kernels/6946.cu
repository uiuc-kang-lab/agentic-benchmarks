#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <limits>

// Combined kernel with dynamic block size and optimized reduction
template <typename scalar_t>
__global__ void argmin_combined_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size,
    int block_size) {

    const int tid = threadIdx.x;
    const int slice_idx = blockIdx.x; // Each block handles one slice
    if (slice_idx >= outer_size * inner_size) return;

    int64_t outer = slice_idx / inner_size;
    int64_t inner = slice_idx % inner_size;

    scalar_t local_min = std::numeric_limits<scalar_t>::max();
    int local_min_idx = 0;

    for (int k = tid; k < K; k += block_size) {
        scalar_t val = __ldg(&x[outer * (static_cast<int64_t>(K) * inner_size) + k * inner_size + inner]);
        if (val < local_min) {
            local_min = val;
            local_min_idx = k;
        }
    }

    extern __shared__ char shared_memory[];
    scalar_t* s_min_vals = reinterpret_cast<scalar_t*>(shared_memory);
    int* s_min_inds = reinterpret_cast<int*>(&s_min_vals[block_size]);

    s_min_vals[tid] = local_min;
    s_min_inds[tid] = local_min_idx;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_min_vals[tid + stride] < s_min_vals[tid]) {
                s_min_vals[tid] = s_min_vals[tid + stride];
                s_min_inds[tid] = s_min_inds[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[slice_idx] = s_min_inds[0];
    }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

    int dims = x.dim();
    if (dim < 0) dim += dims;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }
    int K = static_cast<int>(x.size(dim));
    int64_t inner_size = 1;
    for (int i = dim + 1; i < dims; i++) {
        inner_size *= x.size(i);
    }

    std::vector<int64_t> out_sizes;
    for (int i = 0; i < dims; i++) {
        if (i == dim) continue;
        out_sizes.push_back(x.size(i));
    }
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

    int block_size = (K <= 32) ? 32 : (K <= 64) ? 64 : (K <= 128) ? 128 : (K <= 256) ? 256 : 512;
    int num_slices = outer_size * inner_size;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_combined_kernel<scalar_t><<<num_slices, block_size, 2 * block_size * sizeof(scalar_t)>>>(
            x_data, output_data, K, outer_size, inner_size, block_size);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmin_cuda_forward, "Argmin forward (CUDA)");
}