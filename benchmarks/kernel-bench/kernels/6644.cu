#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cuda_fp16.h>

// Utility function for maximum.
template <typename scalar_t>
__device__ __forceinline__ scalar_t device_max(scalar_t a, scalar_t b) {
    return a > b ? a : b;
}

// Template specialization for the minimum (identity) value for the max reduction.
template <typename scalar_t>
__device__ __forceinline__ scalar_t min_value();

template <>
__device__ __forceinline__ float min_value<float>() { 
    return -FLT_MAX;
}

template <>
__device__ __forceinline__ double min_value<double>() { 
    return -DBL_MAX;
}

template <>
__device__ __forceinline__ __half min_value<__half>() { 
    // The minimum finite half precision value is -65504
    return __half(-65504.0f);
}

// Kernel: Each block computes the maximum over the reduction dimension for one output element using shared memory.
// The reduction is performed in two stages: first, each thread loads a strided portion of the input and computes a local maximum;
// then, a tree reduction in shared memory is performed, using minimal synchronization and warp-level reduction for the final steps.

template <typename scalar_t>
__global__ void shared_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t dim_size,
    const int64_t inner_size
) {
    // Each block processes one output element. The output index combines the outer and inner indices.
    int out_index = blockIdx.x;  
    int outer_idx = out_index / inner_size;
    int inner_idx = out_index % inner_size;
    int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

    // Declare dynamically allocated shared memory
    extern __shared__ char sdata_char[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(sdata_char);

    // Each thread computes a partial maximum over a strided range of the reduction dimension.
    scalar_t local_max = min_value<scalar_t>();
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        int64_t offset = base_offset + i * inner_size;
        scalar_t val = __ldg(input + offset);
        local_max = device_max(local_max, val);
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    // Parallel reduction in shared memory.
    // Reduce using a tree-based approach with __syncthreads() only when necessary,
    // then switch to warp-level reduction for the final steps (no sync needed within a warp).
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = device_max(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        volatile scalar_t* vsdata = sdata;
        if (blockDim.x >= 64) {
            vsdata[threadIdx.x] = device_max(vsdata[threadIdx.x], vsdata[threadIdx.x + 32]);
        }
        if (blockDim.x >= 32) {
            vsdata[threadIdx.x] = device_max(vsdata[threadIdx.x], vsdata[threadIdx.x + 16]);
            vsdata[threadIdx.x] = device_max(vsdata[threadIdx.x], vsdata[threadIdx.x + 8]);
            vsdata[threadIdx.x] = device_max(vsdata[threadIdx.x], vsdata[threadIdx.x + 4]);
            vsdata[threadIdx.x] = device_max(vsdata[threadIdx.x], vsdata[threadIdx.x + 2]);
            vsdata[threadIdx.x] = device_max(vsdata[threadIdx.x], vsdata[threadIdx.x + 1]);
        }
    }

    // Write the final reduced maximum to the output (only thread 0 does this).
    if (threadIdx.x == 0) {
        output[out_index] = sdata[0];
    }
}

// Host function to launch the shared memory reduction kernel.
// Computes outer_size (product of dimensions before 'dim') and inner_size (product of dimensions after 'dim').
// Each block computes one output element over the reduction dimension.

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    // Handle negative dimension
    if (dim < 0) dim += input.dim();

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    int64_t dim_size = input.size(dim);

    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    // Create the output tensor by removing the reduced dimension
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // Determine block size: use a power-of-2 number of threads, up to 256.
    int threads = 256;
    if (dim_size < threads) {
        threads = 1;
        while (threads < dim_size) {
            threads *= 2;
        }
    }

    // Each block computes one output element
    int blocks = outer_size * inner_size;
    size_t shared_mem_bytes = threads * sizeof(at::Half); // default for half types, but will be overwritten in dispatch

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        shared_mem_bytes = threads * sizeof(scalar_t);
        shared_max_reduce_kernel<scalar_t><<<blocks, threads, shared_mem_bytes>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward, "Max reduction forward (CUDA) with shared memory");
}
