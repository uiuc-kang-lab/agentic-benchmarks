#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void matrixMultiplyKernel(const float* __restrict__ A,
                                    float* __restrict__ C,
                                    float s,
                                    int64_t rows,
                                    int64_t cols)
{
    const int tx = threadIdx.x + blockIdx.x * blockDim.x;
    const int ty = threadIdx.y + blockIdx.y * blockDim.y;
    const int stride_x = gridDim.x * blockDim.x;
    const int stride_y = gridDim.y * blockDim.y;

    for (int row = ty; row < rows; row += stride_y) {
        for (int col = tx; col < cols; col += stride_x) {
            const int idx = row * cols + col;
            if (idx < rows * cols) {
                float4* a4_ptr = (float4*)(&A[idx & ~3]);
                float4* c4_ptr = (float4*)(&C[idx & ~3]);
                
                if ((idx & 3) == 0 && idx + 3 < rows * cols) {
                    // Vector load and process when aligned
                    float4 a4 = *a4_ptr;
                    a4.x *= s;
                    a4.y *= s;
                    a4.z *= s;
                    a4.w *= s;
                    *c4_ptr = a4;
                } else {
                    // Single element processing
                    C[idx] = __ldg(&A[idx]) * s;
                }
            }
        }
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    int64_t rows = A.size(0);
    int64_t cols = size / rows;

    const dim3 threads(16, 16);
    const dim3 blocks((cols + threads.x - 1) / threads.x,
                     (rows + threads.y - 1) / threads.y);

    matrixMultiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                             C.data_ptr<float>(),
                                             s,
                                             rows,
                                             cols);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix-scalar multiplication kernel");
}