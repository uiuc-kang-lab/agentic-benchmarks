#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Vectorized type definitions for aligning to 128-bit boundaries

template <typename T>
struct VectorizedType;

// Specialization for float: 4 floats = 128 bits
template <>
struct VectorizedType<float> {
    using type = float4;
    static constexpr int elements = 4;
};

// Specialization for double: 2 doubles = 128 bits
template <>
struct VectorizedType<double> {
    using type = double2;
    static constexpr int elements = 2;
};


// CUDA kernel using __ldg for read-only global memory loads
// and processing data in 128-bit (vectorized) chunks
template <typename scalar_t>
__global__ void relu_kernel_ldg_aligned(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gridSize = blockDim.x * gridDim.x;

    // Determine vector size based on scalar type
    constexpr int vecSize = (sizeof(scalar_t) == sizeof(float)) ? 4 : 
                              ((sizeof(scalar_t) == sizeof(double)) ? 2 : 1);
    
    // Total number of full vectorized loads
    const int totalVectors = size / vecSize;
    const int tailStart = totalVectors * vecSize;

    // Process vectorized portions if vecSize > 1
    if (vecSize > 1) {
        using VecT = typename VectorizedType<scalar_t>::type;
        for (int vecIndex = tid; vecIndex < totalVectors; vecIndex += gridSize) {
            int index = vecIndex * vecSize;
            // Use __ldg for read-only load; reinterpret as vector type
            VecT in_vec = __ldg(reinterpret_cast<const VecT*>(&input[index]));
            VecT out_vec;
            
            // Manually apply ReLU elementwise on the vector
            if (sizeof(scalar_t) == sizeof(float)) {
                out_vec.x = (in_vec.x > 0.f) ? in_vec.x : 0.f;
                out_vec.y = (in_vec.y > 0.f) ? in_vec.y : 0.f;
                out_vec.z = (in_vec.z > 0.f) ? in_vec.z : 0.f;
                out_vec.w = (in_vec.w > 0.f) ? in_vec.w : 0.f;
            } else if (sizeof(scalar_t) == sizeof(double)) {
                out_vec.x = (in_vec.x > 0.0) ? in_vec.x : 0.0;
                out_vec.y = (in_vec.y > 0.0) ? in_vec.y : 0.0;
            }
            // Write the result back (aligned store)
            reinterpret_cast<VecT*>(&output[index])[0] = out_vec;
        }
    }

    // Process tail elements that don't fit in a full vector load
    for (int i = tailStart + tid; i < size; i += gridSize) {
        scalar_t val = __ldg(&input[i]);
        output[i] = (val > 0) ? val : static_cast<scalar_t>(0);
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_kernel_ldg_aligned", ([&] {
        relu_kernel_ldg_aligned<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward with __ldg and 128-bit alignment (CUDA)");
}
