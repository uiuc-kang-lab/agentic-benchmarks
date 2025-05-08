#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

// Generic kernel for non-contiguous accesses or non-float types.
// Uses __ldg() for read-only global memory loads.

template <typename scalar_t>
__global__ void generic_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t num_outputs
) {
    // Each block processes several output elements in a grid-stride loop
    for (int out_idx = blockIdx.x; out_idx < num_outputs; out_idx += gridDim.x) {
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;
        int64_t base = outer_idx * dim_size * inner_size + inner_idx;

        int tid = threadIdx.x;
        int block_size = blockDim.x;
        bool valid = false;
        scalar_t thread_max;
        // Each thread reduces a part of the reduction dimension
        for (int j = tid; j < dim_size; j += block_size) {
            scalar_t val = __ldg(&input[base + j * inner_size]);
            if (!valid) {
                thread_max = val;
                valid = true;
            } else {
                thread_max = max(thread_max, val);
            }
        }

        extern __shared__ char sdata[];
        scalar_t* shmem = reinterpret_cast<scalar_t*>(sdata);
        shmem[tid] = valid ? thread_max : __ldg(&input[base]);
        __syncthreads();

        // Tree-based reduction in shared memory
        for (int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shmem[tid] = max(shmem[tid], shmem[tid + s]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            output[out_idx] = shmem[0];
        }
    }
}

// Vectorized kernel for float type when the reduction dimension is contiguous (inner_size == 1).
// This kernel uses 128-bit aligned loads via float4 and __ldg() to improve global memory throughput.

template <int BLOCK_SIZE>
__global__ void vectorized_float_max_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int64_t dim_size,
    const int64_t num_outputs
) {
    int out_idx = blockIdx.x;
    if (out_idx >= num_outputs) return;

    // For contiguous case, each output element corresponds to a contiguous block of dim_size floats
    const float* start = input + out_idx * dim_size;
    int tid = threadIdx.x;
    int block_size = BLOCK_SIZE;

    // Process in chunks of 4 floats (128 bits)
    int num_vec = dim_size / 4;  // Number of full float4 loads
    int remainder = dim_size % 4; // Remaining elements

    float thread_max = -FLT_MAX;
    for (int i = tid; i < num_vec; i += block_size) {
        const float4* vptr = reinterpret_cast<const float4*>(start);
        float4 data = __ldg(&vptr[i]);
        thread_max = max(thread_max, data.x);
        thread_max = max(thread_max, data.y);
        thread_max = max(thread_max, data.z);
        thread_max = max(thread_max, data.w);
    }

    extern __shared__ float sdata[];
    sdata[tid] = thread_max;
    __syncthreads();

    // Shared memory reduction
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    float block_max = sdata[0];
    // Thread 0 also processes any remaining elements
    if (tid == 0) {
        if (remainder > 0) {
            for (int j = num_vec * 4; j < dim_size; j++) {
                block_max = max(block_max, __ldg(&start[j]));
            }
        }
        output[out_idx] = block_max;
    }
}

// CUDA forward function that selects between the vectorized and generic kernels

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    // Calculate outer_size (product of dimensions before 'dim')
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    // Calculate inner_size (product of dimensions after 'dim')
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    const int64_t dim_size = input.size(dim);
    const int64_t num_outputs = outer_size * inner_size;

    // Prepare output tensor by removing the reduced dimension
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // If the reduction is along a contiguous dimension (inner_size == 1) and type is float,
    // use the vectorized kernel with 128-bit aligned loads.
    if (inner_size == 1 && input.scalar_type() == torch::kFloat) {
        int block_size = 256;
        int grid = num_outputs;  // one block per output element
        size_t shared_mem_size = block_size * sizeof(float);
        vectorized_float_max_reduce_kernel<256><<<grid, block_size, shared_mem_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            dim_size, 
            num_outputs
        );
    } else {
        // Fallback generic kernel using __ldg() for read-only accesses
        int block_size = 256;
        int grid = (num_outputs < 1024) ? num_outputs : 1024;
        size_t shared_mem_size = block_size * input.element_size();
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
            generic_max_reduce_kernel<scalar_t><<<grid, block_size, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size,
                inner_size,
                num_outputs
            );
        }));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward (CUDA) with aligned vectorized loads");
}
