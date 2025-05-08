#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Helper device functions for GELU computation
template <typename scalar_t>
__device__ __forceinline__ scalar_t sqrt2_inv() {
    return static_cast<scalar_t>(0.7071067811865476);  // 1/sqrt(2)
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_erf_input(scalar_t x) {
    return x * sqrt2_inv<scalar_t>();
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_gelu_scale(scalar_t erf_val) {
    return static_cast<scalar_t>(0.5) * (static_cast<scalar_t>(1.0) + erf_val);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t gelu_function(scalar_t x) {
    scalar_t erf_input = compute_erf_input(x);
    scalar_t erf_val = (std::is_same<scalar_t, float>::value) ? 
                       erff(erf_input) : erf(erf_input);
    scalar_t scale = compute_gelu_scale(erf_val);
    return x * scale;
}

template <typename scalar_t>
__global__ void gelu_kernel(const scalar_t* __restrict__ x,
                           scalar_t* __restrict__ y,
                           size_t numel) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < numel; i += stride) {
        y[i] = gelu_function(x[i]);
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(x);
    size_t numel = x.numel();

    const int threads = 256;
    const int blocks = min(65535, (numel + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda", ([&] {
        gelu_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA)");
}