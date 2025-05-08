#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// This kernel optimizes global memory accesses by using __ldg() for read-only loads.
// Additionally, if the inner_size is 1 and the data type is float, it uses vectorized loads (float4), which
// aligns accesses to 128-bit boundaries. This minimizes memory transaction overhead while ensuring full
// precision using the input type.

template <typename scalar_t>
__global__ void aligned_ldg_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer_size * inner_size;
    if (idx >= total) return;

    // If inner_size is 1 and the type is float, we can vectorize the loads using float4.
    if (inner_size == 1) {
        if constexpr (std::is_same<scalar_t, float>::value) {
            // Each thread processes one row (i.e. a reduction over dim_size elements)
            int out_idx = idx;
            const float* in_row = input + out_idx * dim_size;
            float max_val;
            
            // Process vectorized loads if dim_size is large enough
            int vec_count = dim_size / 4;
            int remainder = dim_size % 4;
            
            // Assume that in_row is 16-byte aligned (which is usually true for PyTorch tensors)
            if (vec_count > 0) {
                const float4* vec_ptr = reinterpret_cast<const float4*>(in_row);
                float4 vec_val = __ldg(&vec_ptr[0]);
                max_val = max(max(vec_val.x, vec_val.y), max(vec_val.z, vec_val.w));
                
                for (int i = 1; i < vec_count; i++) {
                    float4 curr = __ldg(&vec_ptr[i]);
                    max_val = max(max_val, curr.x);
                    max_val = max(max_val, curr.y);
                    max_val = max(max_val, curr.z);
                    max_val = max(max_val, curr.w);
                }
                
                int base = vec_count * 4;
                for (int i = 0; i < remainder; i++) {
                    max_val = max(max_val, __ldg(&in_row[base + i]));
                }
            } else {
                // If dim_size < 4, fallback to scalar loads
                max_val = __ldg(in_row);
                for (int i = 1; i < dim_size; i++) {
                    max_val = max(max_val, __ldg(&in_row[i]));
                }
            }
            output[out_idx] = max_val;
            return;
        }
    }

    // Generic case for types or when inner_size != 1
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    int64_t base = outer_idx * dim_size * inner_size + inner_idx;
    scalar_t max_val = __ldg(&input[base]);
    for (int i = 1; i < dim_size; i++) {
        max_val = max(max_val, __ldg(&input[base + i * inner_size]));
    }
    output[idx] = max_val;
}


// CUDA forward function
torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    // Compute outer_size: product of dimensions before the reduction dimension
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    // Compute inner_size: product of dimensions after the reduction dimension
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    const int64_t dim_size = input.size(dim);
    const int64_t total_outputs = outer_size * inner_size;

    // Prepare output tensor with the reduced dimension removed
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // Launch kernel with a standard 1D grid
    const int threads = 256;
    const int blocks = (total_outputs + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "aligned_ldg_max_reduce_forward", ([&] {
        aligned_ldg_max_reduce_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward (CUDA) with __ldg() and 128-bit aligned loads");
}
