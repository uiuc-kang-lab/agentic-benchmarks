#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void multiplyKernel2D(const float* __restrict__ A,
                                float* __restrict__ C,
                                float s,
                                int rows,
                                int cols)
{
    // 2D thread block organization
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;
    
    // Calculate global indices
    const unsigned int row = by * blockDim.y + ty;
    const unsigned int col = bx * blockDim.x + tx;
    
    // Coalesced global memory index
    const unsigned int idx = row * cols + col;
    
    if (row < rows && col < cols) {
        // Use vectorized load when possible (when col is aligned)
        if (col % 4 == 0 && col + 3 < cols) {
            float4 a4 = __ldg(reinterpret_cast<const float4*>(&A[idx]));
            a4.x *= s;
            a4.y *= s;
            a4.z *= s;
            a4.w *= s;
            *reinterpret_cast<float4*>(&C[idx]) = a4;
        } else {
            C[idx] = __ldg(&A[idx]) * s;
        }
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto sizes = A.sizes();
    int64_t rows = sizes[0];
    int64_t cols = sizes.size() > 1 ? sizes[1] : A.numel();
    
    auto C = torch::empty_like(A);
    
    // Use 16x16 thread blocks
    dim3 threads(16, 16);
    dim3 blocks((cols + threads.x - 1) / threads.x,
                (rows + threads.y - 1) / threads.y);

    multiplyKernel2D<<<blocks, threads>>>(A.data_ptr<float>(),
                                         C.data_ptr<float>(),
                                         s,
                                         rows,
                                         cols);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D optimized matrix-scalar multiplication kernel");
}