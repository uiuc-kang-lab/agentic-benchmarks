#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

// Device function for contiguous reduction using vectorized loads when possible
template <typename scalar_t>
__device__ inline scalar_t reduce_sum_contiguous(const scalar_t* data, int dim_size) {
    scalar_t sum = static_cast<scalar_t>(0);
    if constexpr (std::is_same<scalar_t, float>::value) {
        // Use float4 for 128-bit aligned vectorized loads if possible
        if ((dim_size % 4 == 0) && ((((uintptr_t)data) & 0xF) == 0)) {
            int vec_iters = dim_size / 4;
            for (int i = 0; i < vec_iters; i++) {
                float4 val = __ldg(reinterpret_cast<const float4*>(data) + i);
                sum += val.x + val.y + val.z + val.w;
            }
        } else {
            for (int i = 0; i < dim_size; i++) {
                sum += __ldg(data + i);
            }
        }
    } else if constexpr (std::is_same<scalar_t, double>::value) {
        // Use double2 for 16-byte vectorized loads if possible
        if ((dim_size % 2 == 0) && ((((uintptr_t)data) & 0xF) == 0)) {
            int vec_iters = dim_size / 2;
            for (int i = 0; i < vec_iters; i++) {
                double2 val = __ldg(reinterpret_cast<const double2*>(data) + i);
                sum += val.x + val.y;
            }
        } else {
            for (int i = 0; i < dim_size; i++) {
                sum += __ldg(data + i);
            }
        }
    } else {
        // Fallback to scalar loads for other types
        for (int i = 0; i < dim_size; i++) {
            sum += __ldg(data + i);
        }
    }
    return sum;
}

// Device function for strided reduction (non-contiguous memory access)
template <typename scalar_t>
__device__ inline scalar_t reduce_sum_strided(const scalar_t* data, int dim_size, int stride) {
    scalar_t sum = static_cast<scalar_t>(0);
    for (int i = 0; i < dim_size; i++) {
        sum += __ldg(data + i * stride);
    }
    return sum;
}

// Main kernel using modular device functions
template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {
    extern __shared__ scalar_t sdata[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    if (tid >= outer_size * inner_size) return;

    int outer_idx = tid / inner_size;
    int inner_idx = tid % inner_size;
    int64_t offset = outer_idx * (dim_size * inner_size) + inner_idx;

    scalar_t sum;
    if (inner_size == 1) {
        sum = reduce_sum_contiguous(input + offset, dim_size);
    } else {
        sum = reduce_sum_strided(input + offset, dim_size, inner_size);
    }

    // Use shared memory for block reduction
    sdata[local_tid] = sum;
    __syncthreads();

    // Reduce within the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_tid < s) {
            sdata[local_tid] += sdata[local_tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (local_tid == 0) {
        atomicAdd(&output[outer_idx * inner_size + inner_idx], sdata[0] / static_cast<scalar_t>(dim_size));
    }
}

// Host function launching the kernel
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    // Calculate outer and inner sizes
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Prepare output tensor by removing the reduction dimension
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
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
    m.def("forward", &mean_reduce_cuda, "Mean reduction with atomic operations (CUDA)");
}
