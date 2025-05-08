#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ float4 loadAndMultiply(const float* __restrict__ A, float s, int idx) {
    float4 a4 = __ldg(reinterpret_cast<const float4*>(&A[idx * 4]));
    a4.x *= s;
    a4.y *= s;
    a4.z *= s;
    a4.w *= s;
    return a4;
}

__device__ void storeResult(float* __restrict__ C, float4 result, int idx) {
    float4* c4_ptr = reinterpret_cast<float4*>(&C[idx * 4]);
    *c4_ptr = result;
}

__global__ void multiplyKernelModular(const float* __restrict__ A,
                                       float* __restrict__ C,
                                       float s,
                                       int64_t size)
{
    const unsigned int totalThreads = gridDim.x * blockDim.x;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int numVec = size / 4;  // number of groups of 4 elements

    // Process groups of 4 elements in a loop to improve occupancy and avoid divergence
    for (int i = tid; i < numVec; i += totalThreads) {
        float4 result = loadAndMultiply(A, s, i);
        storeResult(C, result, i);
    }

    // Handle any remaining tail elements
    int tailStart = numVec * 4;
    for (int i = tailStart + tid; i < size; i += totalThreads) {
        C[i] = __ldg(&A[i]) * s;
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

    multiplyKernelModular<<<blocks, threads>>>(A.data_ptr<float>(),
                                               C.data_ptr<float>(),
                                               s,
                                               size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular matrix-scalar multiplication kernel");
}