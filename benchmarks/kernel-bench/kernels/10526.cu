#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<typename T>
struct VectorType {};

template<>
struct VectorType<float> {
    using type = float4;
    static constexpr int size = 4;
};

__global__ void cumsum_vec4_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int outer_size,
                                   int inner_size,
                                   int stride) {
    using vec_t = VectorType<float>::type;
    constexpr int vec_size = VectorType<float>::size;
    
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x * vec_size;
    
    if (outer_idx < outer_size && inner_idx < inner_size - vec_size + 1) {
        float sum[vec_size] = {0};
        int base = outer_idx * stride * inner_size;
        
        for (int i = 0; i < stride; ++i) {
            const vec_t* vin = reinterpret_cast<const vec_t*>(
                &input[base + i * inner_size + inner_idx]
            );
            vec_t vsum;

            vsum.x = sum[0] += vin->x;
            vsum.y = sum[1] += vin->y;
            vsum.z = sum[2] += vin->z;
            vsum.w = sum[3] += vin->w;
            
            *reinterpret_cast<vec_t*>(&output[base + i * inner_size + inner_idx]) = vsum;
        }
    }
}

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;

    int outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= x.size(i);

    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) inner_size *= x.size(i);

    int stride = x.size(dim);
    
    dim3 blocks(outer_size);
    dim3 threads((inner_size + VectorType<float>::size - 1) / VectorType<float>::size);
    
    cumsum_vec4_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                          output.data_ptr<float>(),
                                          outer_size,
                                          inner_size,
                                          stride);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized CUDA cumulative sum");
}