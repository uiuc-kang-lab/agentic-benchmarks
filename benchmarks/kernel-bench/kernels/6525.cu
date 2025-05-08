#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void scalar_mean_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < outer_size * inner_size) {
        int outer_idx = tid / inner_size;
        int inner_idx = tid % inner_size;
        int input_offset = outer_idx * dim_size * inner_size + inner_idx;
        
        scalar_t sum = 0;
        for (int i = 0; i < dim_size; i++) {
            sum += input[input_offset + i * inner_size];
        }
        output[tid] = sum / dim_size;
        tid += gridDim.x * blockDim.x;
    }
}

__global__ void vector_mean_kernel_float4(
    const float* input,
    float* output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_vec) {
    
    constexpr int VEC_SIZE = 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < outer_size * inner_vec) {
        int outer_idx = tid / inner_vec;
        int vec_idx = tid % inner_vec;
        int input_base = (outer_idx * dim_size * inner_vec + vec_idx) * VEC_SIZE;
        
        float4 sum = make_float4(0, 0, 0, 0);
        for (int i = 0; i < dim_size; i++) {
            const float4* vec_ptr = reinterpret_cast<const float4*>(input + input_base + i * inner_vec * VEC_SIZE);
            float4 val = *vec_ptr;
            sum.x += val.x;
            sum.y += val.y;
            sum.z += val.z;
            sum.w += val.w;
        }
        
        sum.x /= dim_size;
        sum.y /= dim_size;
        sum.z /= dim_size;
        sum.w /= dim_size;
        
        float4* out_ptr = reinterpret_cast<float4*>(output + outer_idx * inner_vec * VEC_SIZE + vec_idx * VEC_SIZE);
        *out_ptr = sum;
        tid += gridDim.x * blockDim.x;
    }
}

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++)
        outer_size *= sizes[i];
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++)
        inner_size *= sizes[i];
    
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());
    
    const int threads = 256;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", [&] {
        constexpr int VEC_SIZE = 4;
        if (std::is_same<scalar_t, float>::value && (inner_size % VEC_SIZE) == 0) {
            int64_t inner_vec = inner_size / VEC_SIZE;
            int blocks = (outer_size * inner_vec + threads - 1) / threads;
            vector_mean_kernel_float4<<<blocks, threads>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                outer_size,
                dim_size,
                inner_vec
            );
        } else {
            int blocks = (outer_size * inner_size + threads - 1) / threads;
            scalar_mean_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer_size,
                dim_size,
                inner_size
            );
        }
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Vectorized mean reduction (CUDA)");
}