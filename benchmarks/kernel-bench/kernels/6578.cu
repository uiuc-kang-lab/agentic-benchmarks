#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Generic kernel that uses __ldg() for read-only global memory loads.
// This kernel works for any inner_size and relies on the assumption that the input
// tensor is properly aligned. __restrict__ is used to enable compiler optimizations.

template <typename scalar_t>
__global__ void max_reduce_aligned_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = outer_size * inner_size;
    if (idx >= total_elements) return;

    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    // Compute the starting index. We assume that the input pointer is aligned
    // to 128-bit boundaries when possible, which aids coalesced accesses.
    const int64_t base = outer_idx * dim_size * inner_size + inner_idx;

    // Use __ldg() for read-only loads
    scalar_t max_val = __ldg(input + base);
    for (int i = 1; i < dim_size; i++) {
        scalar_t val = __ldg(input + base + i * inner_size);
        max_val = max(max_val, val);
    }
    output[idx] = max_val;
}

// Specialized kernel for the case when inner_size == 1 and the type is float.
// In this scenario the data is contiguous along the reduction dimension, and we
// can use vectorized loads (float4) to load 128 bits (4 floats) at a time. This
// ensures that the global memory loads are aligned to 128-bit boundaries.

__global__ void max_reduce_vectorized_kernel_float(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_size) return;

    // Each thread processes one row (since inner_size == 1).
    const float* row = input + idx * dim_size;
    float max_val;
    int i = 0;
    const int vector_len = 4; // 128 bits / 32 bits per float

    // Use vectorized load if there are at least vector_len elements
    if (dim_size >= vector_len) {
        // Assume the pointer 'row' is 16-byte aligned.
        float4 vec = *reinterpret_cast<const float4*>(row);
        max_val = max(max(max(vec.x, vec.y), vec.z), vec.w);
        i = vector_len;
    } else {
        max_val = row[0];
        i = 1;
    }

    // Continue with scalar loads using __ldg()
    for (; i < dim_size; i++) {
        float val = __ldg(row + i);
        max_val = max(max_val, val);
    }
    output[idx] = max_val;
}

// CUDA forward function which selects between the vectorized and generic kernel based
// on input characteristics. If the inner_size is 1 and the type is float, the
// vectorized kernel is launched to benefit from aligned 128-bit loads.

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    const int64_t dim_size = input.size(dim);

    // Prepare output tensor by removing the reduced dimension
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // If the reduction is over a contiguous dimension (inner_size == 1) and the type is float,
    // we can use the vectorized kernel to load 128 bits at a time.
    if (inner_size == 1 && input.scalar_type() == torch::kFloat) {
        const int threads = 256;
        const int blocks = (outer_size + threads - 1) / threads;
        max_reduce_vectorized_kernel_float<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            outer_size,
            dim_size
        );
    } else {
        const int threads = 256;
        const int blocks = ((outer_size * inner_size) + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_aligned_forward", ([&] {
            max_reduce_aligned_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer_size,
                dim_size,
                inner_size
            );
        }));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward optimized using __ldg() and aligned 128-bit loads");
}
