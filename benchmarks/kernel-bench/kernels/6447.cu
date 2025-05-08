#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

template<typename T>
struct VectorType {};

template<>
struct VectorType<float> {
    using vec_t = float4;
    static constexpr int vec_size = 4;
};

template<>
struct VectorType<double> {
    using vec_t = double2;
    static constexpr int vec_size = 2;
};

template <typename scalar_t>
__device__ __forceinline__ scalar_t reduce_sum_vectorized(
    const scalar_t* __restrict__ data,
    const int64_t dim_size,
    const int64_t stride) {
    
    scalar_t sum = 0;
    
    if (stride == 1) {
        const bool is_aligned = ((uintptr_t)data & 0xF) == 0;
        
        if constexpr (std::is_same<scalar_t, float>::value || std::is_same<scalar_t, double>::value) {
            using Vector = typename VectorType<scalar_t>::vec_t;
            constexpr int vec_size = VectorType<scalar_t>::vec_size;
            
            if (is_aligned && dim_size >= vec_size && dim_size % vec_size == 0) {
                const Vector* vec_data = reinterpret_cast<const Vector*>(data);
                const int vec_iters = dim_size / vec_size;
                
                #pragma unroll 4
                for (int i = 0; i < vec_iters; i++) {
                    Vector val = __ldg(vec_data + i);
                    if constexpr (std::is_same<scalar_t, float>::value)
                        sum += val.x + val.y + val.z + val.w;
                    else
                        sum += val.x + val.y;
                }
                return sum;
            }
        }
    }
    
    #pragma unroll 4
    for (int i = 0; i < dim_size; i++) {
        sum += __ldg(data + i * stride);
    }
    return sum;
}

template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;
    
    const int outer_idx = tid / inner_size;
    const int inner_idx = tid % inner_size;
    const int64_t offset = outer_idx * dim_size * inner_size + inner_idx;
    
    const scalar_t sum = reduce_sum_vectorized(
        input + offset,
        dim_size,
        inner_size
    );
    
    output[tid] = sum / static_cast<scalar_t>(dim_size);
}

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    const int64_t dim_size = sizes[dim];
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= sizes[i];
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) inner_size *= sizes[i];
    
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());
    
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));
    
    return output;
}