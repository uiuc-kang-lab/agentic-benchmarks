#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void aligned_mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;

    const int outer_idx = tid / inner_size;
    const int inner_idx = tid % inner_size;
    const int64_t input_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    scalar_t sum = 0;

    if constexpr (std::is_same<scalar_t, float>::value) {
        // Check if input pointer is aligned and dimension size is multiple of 4
        const bool is_aligned = (((uintptr_t)(input + input_offset) & 15) == 0) && 
                              (inner_size % 4 == 0);
        
        if (is_aligned && dim_size >= 4) {
            // Use vectorized loads for aligned data
            const float4* input_vec4 = reinterpret_cast<const float4*>(input + input_offset);
            const int vec4_steps = dim_size / 4;
            
            #pragma unroll 4
            for (int i = 0; i < vec4_steps; i++) {
                float4 v = __ldg(&input_vec4[i * inner_size/4]);
                sum += v.x + v.y + v.z + v.w;
            }

            // Handle remaining elements
            #pragma unroll
            for (int i = vec4_steps * 4; i < dim_size; i++) {
                sum += __ldg(input + input_offset + i * inner_size);
            }
        } else {
            // Fallback for unaligned data
            #pragma unroll 4
            for (int i = 0; i < dim_size; i++) {
                sum += __ldg(input + input_offset + i * inner_size);
            }
        }
    } else if constexpr (std::is_same<scalar_t, double>::value) {
        // Similar optimization for double precision using double2
        const bool is_aligned = (((uintptr_t)(input + input_offset) & 15) == 0) && 
                              (inner_size % 2 == 0);
        
        if (is_aligned && dim_size >= 2) {
            const double2* input_vec2 = reinterpret_cast<const double2*>(input + input_offset);
            const int vec2_steps = dim_size / 2;
            
            #pragma unroll 4
            for (int i = 0; i < vec2_steps; i++) {
                double2 v = __ldg(&input_vec2[i * inner_size/2]);
                sum += v.x + v.y;
            }

            #pragma unroll
            for (int i = vec2_steps * 2; i < dim_size; i++) {
                sum += __ldg(input + input_offset + i * inner_size);
            }
        } else {
            #pragma unroll 4
            for (int i = 0; i < dim_size; i++) {
                sum += __ldg(input + input_offset + i * inner_size);
            }
        }
    } else {
        // Fallback for other types
        #pragma unroll 4
        for (int i = 0; i < dim_size; i++) {
            sum += __ldg(input + input_offset + i * inner_size);
        }
    }

    output[tid] = sum / static_cast<scalar_t>(dim_size);
}

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    const int64_t dim_size = sizes[dim];
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }
    
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());
    
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "aligned_mean_reduce_cuda", ([&] {
        aligned_mean_reduce_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &mean_reduce_cuda, "Aligned vectorized mean reduction (CUDA)");
}