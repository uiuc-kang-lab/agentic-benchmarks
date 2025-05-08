#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void multiplyKernel(const float* __restrict__ A,
                               float* __restrict__ C,
                               float s,
                               int64_t size)
{
    const unsigned int warp_size = 32;
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_id = tid / warp_size;
    const unsigned int num_complete_warps = size / warp_size;
    
    if (warp_id < num_complete_warps) {
        // Complete warps - no divergence
        C[tid] = A[tid] * s;
    }
    else if (warp_id == num_complete_warps) {
        // Handle remaining elements in last partial warp
        const unsigned int remaining = size % warp_size;
        const unsigned int lane_id = tid % warp_size;
        if (lane_id < remaining) {
            C[tid] = A[tid] * s;
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
    const int blocks = (size + threads - 1) / threads;

    multiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                       C.data_ptr<float>(),
                                       s,
                                       size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix-scalar multiplication kernel");
}