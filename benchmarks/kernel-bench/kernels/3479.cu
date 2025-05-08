#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x);

template <>
__device__ inline float gelu_function<float>(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

template <>
__device__ inline double gelu_function<double>(double x) {
    return x * 0.5 * (1.0 + erf(x / 1.4142135623730951));
}

template <typename scalar_t, int VEC_SIZE>
__global__ void gelu_kernel(const scalar_t* __restrict__ x,
                            scalar_t* __restrict__ y,
                            size_t numel) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    using VecType = scalar_t __attribute__((ext_vector_type(VEC_SIZE)));
    const VecType* x_vec = reinterpret_cast<const VecType*>(x);
    VecType* y_vec = reinterpret_cast<VecType*>(y);
    
    for (int i = tid; i < numel/VEC_SIZE; i += stride) {
        VecType vec = x_vec[i];
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            vec[j] = gelu_function(vec[j]);
        }
        y_vec[i] = vec;
    }

    // Handle remaining elements
    const int remainder_start = (numel/VEC_SIZE)*VEC_SIZE;
    for (int i = remainder_start + tid; i < numel; i += stride) {
        y[i] = gelu_function(x[i]);
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(x);
    size_t numel = x.numel();

    constexpr int BLOCK_SIZE = 256;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda", ([&] {
        constexpr int vec_size = std::is_same<scalar_t, float>::value ? 4 : 2;
        int num_blocks = (numel + BLOCK_SIZE * vec_size - 1) / (BLOCK_SIZE * vec_size);
        num_blocks = min(num_blocks, 144 * 4);  // 144 SMs * 4 waves
        gelu_kernel<scalar_t, vec_size><<<num_blocks, BLOCK_SIZE>>>(x.data_ptr<scalar_t>(),
                                                                    output.data_ptr<scalar_t>(),
                                                                    numel);
    }));

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda", ([&] {
        gelu_kernel<scalar_t, VEC_SIZE><<<num_blocks, BLOCK_SIZE>>>(x.data_ptr<scalar_t>(),
                                                                   output.data_ptr<scalar_t>(),
                                                                   numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA)");
}