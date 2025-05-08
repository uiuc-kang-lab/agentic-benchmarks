#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>

template <typename scalar_t, int VEC_SIZE>
__global__ void hardsigmoid_vector_kernel(const scalar_t* __restrict__ input,
                                        scalar_t* __restrict__ output,
                                        size_t numel) {
    constexpr scalar_t three = static_cast<scalar_t>(3.0);
    constexpr scalar_t sixth = static_cast<scalar_t>(1.0/6.0);
    
    using vec_t = typename std::conditional<
        std::is_same<scalar_t, double>::value, double2,
        typename std::conditional<std::is_same<scalar_t, float>::value, float4, half2>::type
    >::type;
    
    const int tid = blockIdx.x * (blockDim.x * VEC_SIZE) + threadIdx.x * VEC_SIZE;
    
    if (tid + VEC_SIZE <= numel) {
        vec_t chunk = *reinterpret_cast<const vec_t*>(&input[tid]);
        scalar_t elements[VEC_SIZE];
        *reinterpret_cast<vec_t*>(elements) = chunk;
        
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            elements[i] = max(static_cast<scalar_t>(0.0),
                            min(static_cast<scalar_t>(1.0),
                                (elements[i] + three) * sixth));
        }
        
        *reinterpret_cast<vec_t*>(&output[tid]) = *reinterpret_cast<vec_t*>(elements);
    }
    else {  // Handle residual elements
        for (int i = 0; i < VEC_SIZE && tid + i < numel; ++i) {
            scalar_t a = input[tid + i];
            output[tid + i] = max(static_cast<scalar_t>(0.0),
                                min(static_cast<scalar_t>(1.0),
                                    (a + three) * sixth));
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();
    
    const int vec_size = input.dtype() == torch::kFloat16 ? 2 : 4;
    const int threads = 256;
    const int elements_per_block = threads * vec_size;
    const int blocks = (numel + elements_per_block - 1) / elements_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_cuda", ([&] {
        if (vec_size == 4) {
            hardsigmoid_vector_kernel<scalar_t, 4><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                numel);
        } else {
            hardsigmoid_vector_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                numel);
        }
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid vectorized forward (CUDA)");
}