#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

///////////////////////////////////////
// Modular Device Functions Section
///////////////////////////////////////

// Scalar ReLU function for both float and double
template <typename scalar_t>
__device__ __forceinline__ scalar_t relu_scalar(scalar_t val) {
    return val > static_cast<scalar_t>(0) ? val : static_cast<scalar_t>(0);
}

// Vectorized ReLU for float4: processes 4 floats at once
__device__ __forceinline__ float4 relu_vectorized_float4(float4 val) {
    float4 res;
    res.x = relu_scalar(val.x);
    res.y = relu_scalar(val.y);
    res.z = relu_scalar(val.z);
    res.w = relu_scalar(val.w);
    return res;
}

// Vectorized ReLU for double2: processes 2 doubles at once
__device__ __forceinline__ double2 relu_vectorized_double2(double2 val) {
    double2 res;
    res.x = relu_scalar(val.x);
    res.y = relu_scalar(val.y);
    return res;
}

///////////////////////////////////
// Modular CUDA Kernel Section
///////////////////////////////////

// This kernel uses a grid-stride loop and applies vectorized operations where possible
// It leverages modular device functions for both scalar and vectorized ReLU computations

template <typename scalar_t>
__global__ void modular_relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    if constexpr (sizeof(scalar_t) == 4) {  // For float type using float4 vectorization
        int vec_count = size / 4;  // Number of float4 vectors
        float4* out_vec = reinterpret_cast<float4*>(output);
        const float4* in_vec = reinterpret_cast<const float4*>(input);

        // Process the vectorized portion
        for (int i = idx; i < vec_count; i += stride) {
            out_vec[i] = relu_vectorized_float4(in_vec[i]);
        }

        // Process remaining elements
        for (int i = vec_count * 4 + idx; i < size; i += stride) {
            output[i] = relu_scalar(input[i]);
        }
    } else {  // For double type using double2 vectorization
        int vec_count = size / 2;  // Number of double2 vectors
        double2* out_vec = reinterpret_cast<double2*>(output);
        const double2* in_vec = reinterpret_cast<const double2*>(input);

        // Process the vectorized portion
        for (int i = idx; i < vec_count; i += stride) {
            out_vec[i] = relu_vectorized_double2(in_vec[i]);
        }

        // Process remaining elements if any
        for (int i = vec_count * 2 + idx; i < size; i += stride) {
            output[i] = relu_scalar(input[i]);
        }
    }
}

//////////////////////////////////
// Pybind11 Entry Point Section
//////////////////////////////////

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "modular_relu_kernel", ([&] {
        modular_relu_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular ReLU with device functions (CUDA)");
}
