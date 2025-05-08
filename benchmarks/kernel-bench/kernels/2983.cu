#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>

template <typename T, int VEC_SIZE> struct VectorType {};
template <> struct VectorType<float, 4> { using type = float4; };
template <> struct VectorType<double, 2> { using type = double2; };

template <typename T>
__device__ T compute_tanh(T x) { return ::tanh(x); }

template <typename VecType>
__device__ VecType tanh_vec(const VecType& vec_in) {
    VecType vec_out;
    vec_out.x = compute_tanh(vec_in.x);
    vec_out.y = compute_tanh(vec_in.y);
    if constexpr (sizeof(VecType) == sizeof(float4)) {
        vec_out.z = compute_tanh(vec_in.z);
        vec_out.w = compute_tanh(vec_in.w);
    }
    return vec_out;
}

template <typename T>
__global__ void vectorized_tanh(const T* input, T* output, int size) {
    constexpr int VEC_SIZE = sizeof(typename VectorType<T, 4>::type)/sizeof(T);
    using VecType = typename VectorType<T, VEC_SIZE>::type;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_size = static_cast<int>(size / VEC_SIZE);
    
    // Vectorized processing
    for (int i = tid; i < vec_size; i += gridDim.x * blockDim.x) {
        VecType in = *reinterpret_cast<const VecType*>(input + i*VEC_SIZE);
        *reinterpret_cast<VecType*>(output + i*VEC_SIZE) = tanh_vec(in);
    }
    
    // Remainder elements
    for (int i = vec_size*VEC_SIZE + tid; i < size; i += gridDim.x * blockDim.x) {
        output[i] = compute_tanh(input[i]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "vectorized_tanh", [&] {
        vectorized_tanh<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized Tanh forward (CUDA)");
}
