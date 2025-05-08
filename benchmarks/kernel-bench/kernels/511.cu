#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void multiplyKernelStreamed(const float* __restrict__ A,
                                        float* __restrict__ C,
                                        float s,
                                        int64_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * s;
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    multiplyKernelStreamed<<<blocks, threads, 0, stream>>>(A.data_ptr<float>(),
                                                          C.data_ptr<float>(),
                                                          s,
                                                          size);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix-scalar multiplication kernel with CUDA streams");
}