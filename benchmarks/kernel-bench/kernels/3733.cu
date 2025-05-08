#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int VEC_SIZE>
__global__ void hardsigmoid_vec_warp_kernel(const scalar_t* __restrict__ input,
                                          scalar_t* __restrict__ output,
                                          size_t numel) {
    constexpr scalar_t sixth = static_cast<scalar_t>(1.0/6.0);
    constexpr scalar_t half = static_cast<scalar_t>(0.5);
    
    using vec_t = typename std::conditional<
        std::is_same<scalar_t, float>::value, float4,
        typename std::conditional<std::is_same<scalar_t, double>::value, double2, void>::type
    >::type;

    const int num_vec = numel / VEC_SIZE;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;

    // Vector processing
    for(int vec_idx = tid; vec_idx < num_vec; vec_idx += total_threads) {
        vec_t chunk;
        chunk = *reinterpret_cast<const vec_t*>(input + vec_idx*VEC_SIZE);
        scalar_t tmp[VEC_SIZE];
        *reinterpret_cast<vec_t*>(tmp) = chunk;
        
        #pragma unroll
        for(int i=0; i<VEC_SIZE; ++i) {
            tmp[i] = fma(tmp[i], sixth, half); // (x + 3)/6
            tmp[i] = fmax(static_cast<scalar_t>(0.0), fmin(static_cast<scalar_t>(1.0), tmp[i]));
        }
        *reinterpret_cast<vec_t*>(output + vec_idx*VEC_SIZE) = *reinterpret_cast<vec_t*>(tmp);
    }

    // Residual processing
    const int scalar_start = num_vec * VEC_SIZE;
    for(int i = scalar_start + tid; i < numel; i += total_threads) {
        scalar_t x = input[i];
        x = fma(x, sixth, half);
        x = fmax(static_cast<scalar_t>(0.0), fmin(static_cast<scalar_t>(1.0), x));
        output[i] = x;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();
    
    const int threads = 512;
    const int vec_size = (input.dtype() == torch::kFloat32) ? 4 : 2;
    const int blocks = (numel / vec_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_vec_warp", ([&] {
        hardsigmoid_vec_warp_kernel<scalar_t, vec_size><<<blocks, threads>>>(
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
    m.def("forward", &forward, "HardSigmoid optimized vectorized warp forward (CUDA)");
}