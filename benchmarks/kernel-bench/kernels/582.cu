#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace {

__device__ __inline__ float4 load_vectorized(const float4* ptr, int idx) {
    return ptr[idx];
}

__device__ __inline__ float4 multiply_element(float4 val, float scalar) {
    val.x *= scalar;
    val.y *= scalar;
    val.z *= scalar;
    val.w *= scalar;
    return val;
}

__device__ __inline__ void store_vectorized(float4* ptr, int idx, float4 val) {
    ptr[idx] = val;
}

__global__ void vectorized_multiply_kernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    float scalar,
    int64_t num_vectors) {
    
    constexpr int elements_per_thread = 4;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for(int i = tid; i < num_vectors; i += stride) {
        float4 data = load_vectorized(input, i);
        data = multiply_element(data, scalar);
        store_vectorized(output, i, data);
    }
}

__global__ void scalar_multiply_remainder(
    const float* __restrict__ input,
    float* __restrict__ output,
    float scalar,
    int64_t start_index,
    int64_t total_elements) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int element = start_index + tid;
    
    if(element < total_elements) {
        output[element] = input[element] * scalar;
    }
}

} // anonymous namespace

torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(A.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "Input must be float32");

    auto C = torch::empty_like(A);
    const int64_t total_elements = A.numel();
    const int64_t vectorized_elements = total_elements / 4;
    const int64_t remainder = total_elements % 4;

    constexpr int threads_per_block = 256;
    const int blocks = (vectorized_elements + threads_per_block - 1) / threads_per_block;

    if(vectorized_elements > 0) {
        vectorized_multiply_kernel<<<blocks, threads_per_block>>>(
            reinterpret_cast<const float4*>(A.data_ptr<float>()),
            reinterpret_cast<float4*>(C.data_ptr<float>()),
            s,
            vectorized_elements
        );
    }

    if(remainder > 0) {
        const int64_t remainder_start = vectorized_elements * 4;
        const int remainder_blocks = (remainder + threads_per_block - 1) / threads_per_block;
        
        scalar_multiply_remainder<<<remainder_blocks, threads_per_block>>>(
            A.data_ptr<float>(),
            C.data_ptr<float>(),
            s,
            remainder_start,
            total_elements
        );
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized modular matrix-scalar multiplication");
}