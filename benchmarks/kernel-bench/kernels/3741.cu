#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename scalar_t, int VEC_SIZE>
__global__ void hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   size_t n) {
    constexpr scalar_t three = 3.0;
    constexpr scalar_t sixth = 1.0/6.0;
    
    using vec_t = typename std::conditional<
        std::is_same<scalar_t, float>::value, float4,
        typename std::conditional<std::is_same<scalar_t, double>::value, double2, void>::type
    >::type;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Process vectorized elements
    for(int i = tid * VEC_SIZE; i < n; i += stride * VEC_SIZE) {
        vec_t vec_in;
        scalar_t elements[VEC_SIZE];
        
        if(i + VEC_SIZE <= n) {
            vec_in = *reinterpret_cast<const vec_t*>(&input[i]);
            *reinterpret_cast<vec_t*>(elements) = vec_in;

            #pragma unroll
            for(int v = 0; v < VEC_SIZE; v++) {
                elements[v] = fmaxf(0.0f, fminf(1.0f, fma(elements[v], sixth, three * sixth)));
            }

            *reinterpret_cast<vec_t*>(&output[i]) = *reinterpret_cast<vec_t*>(elements);
        }
    }

    // Process remaining elements
    for(int i = tid; i < n; i += stride) {
        if(i >= (n / VEC_SIZE) * VEC_SIZE) {
            scalar_t x = input[i];
            x = fmaxf(0.0f, fminf(1.0f, fma(x, sixth, three * sixth)));
            output[i] = x;
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();

    const int vec_size = sizeof(float4)/sizeof(float);
    const int threads = 256;
    const int blocks = std::min((numel + threads * vec_size - 1) / (threads * vec_size), 1024);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_cuda", ([&] {
        hardsigmoid_kernel<scalar_t, vec_size><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid activation forward (CUDA)");
}
