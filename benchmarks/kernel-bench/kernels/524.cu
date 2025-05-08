#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Device function to handle vectorized load and multiply
__device__ inline float4 load_and_multiply(const float* __restrict__ A, float s, int idx) {
    float4 a4 = __ldg(reinterpret_cast<const float4*>(&A[idx * 4]));
    a4.x *= s;
    a4.y *= s;
    a4.z *= s;
    a4.w *= s;
    return a4;
}

// Device function to store result back to global memory
__device__ inline void store_result(float* __restrict__ C, float4 a4, int idx) {
    reinterpret_cast<float4*>(&C[idx * 4])[0] = a4;
}

__global__ void modularMultiplyKernel(const float* __restrict__ A,
                                       float* __restrict__ C,
                                       float s,
                                       int64_t size)
{
    // Thread index
    int vecCount = size / 4;
    int remainder = size % 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < vecCount) {
        float4 a4 = load_and_multiply(A, s, tid);
        store_result(C, a4, tid);
    }

    int tailStart = vecCount * 4;
    int tailIdx = tid - vecCount;
    if (tailIdx < remainder) {
        int index = tailStart + tailIdx;
        C[index] = __ldg(&A[index]) * s;
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    const int threads = 256;
    const int blocks = (size + threads * 4 - 1) / (threads * 4);

    modularMultiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                                C.data_ptr<float>(),
                                                s,
                                                size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular Matrix-Scalar Multiplication Kernel");
}