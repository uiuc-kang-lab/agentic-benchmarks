#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void multiplyKernelUnroll(const float* __restrict__ A,
                                      float* __restrict__ C,
                                      float s,
                                      int64_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        int current_idx = idx * 4 + i;
        if (current_idx < size) {
            C[current_idx] = A[current_idx] * s;
        }
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    const int threads = 256;
    // Calculate grid size to exactly match the data size with the unrolling factor
    const int elements_per_thread = 4;
    const int total_threads_needed = (size + elements_per_thread - 1) / elements_per_thread;
    const int blocks = (total_threads_needed + threads - 1) / threads;

    multiplyKernelUnroll<<<blocks, threads>>>(A.data_ptr<float>(),
                                              C.data_ptr<float>(),
                                              s,
                                              size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix-scalar multiplication kernel with loop unrolling");
}