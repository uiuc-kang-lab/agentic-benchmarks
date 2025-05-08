#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

template<typename T>
__device__ __forceinline__ T load_vector(const float* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const T*>(ptr));
}

template<typename T>
__device__ __forceinline__ void store_vector(float* __restrict__ ptr, T val) {
    *reinterpret_cast<T*>(ptr) = val;
}

__device__ __forceinline__ float4 compute_multiplication(float4 input, float scalar) {
    float4 result;
    result.x = input.x * scalar;
    result.y = input.y * scalar;
    result.z = input.z * scalar;
    result.w = input.w * scalar;
    return result;
}

__device__ __forceinline__ void process_vector4(const float* __restrict__ A,
                                               float* __restrict__ C,
                                               float s,
                                               int idx) {
    float4 input = load_vector<float4>(&A[idx]);
    float4 result = compute_multiplication(input, s);
    store_vector<float4>(&C[idx], result);
}

__device__ __forceinline__ void process_scalar(const float* __restrict__ A,
                                             float* __restrict__ C,
                                             float s,
                                             int idx) {
    C[idx] = __ldg(&A[idx]) * s;
}

__global__ void multiplyKernelModular(const float* __restrict__ A,
                                     float* __restrict__ C,
                                     float s,
                                     int64_t size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vector_size = 4;
    
    // Process aligned elements with float4
    const int aligned_size = size & ~(vector_size - 1);
    
    for (int idx = tid * vector_size; idx < aligned_size; idx += stride * vector_size) {
        process_vector4(A, C, s, idx);
    }
    
    // Handle remaining elements
    for (int idx = aligned_size + tid; idx < size; idx += stride) {
        process_scalar(A, C, s, idx);
    }
}

torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    const int threads = 256;
    const int vector_size = 4;
    const int blocks = std::min(256, static_cast<int>((size + threads * vector_size - 1) / (threads * vector_size)));

    multiplyKernelModular<<<blocks, threads>>>(A.data_ptr<float>(),
                                             C.data_ptr<float>(),
                                             s,
                                             size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular optimized matrix-scalar multiplication kernel");
}