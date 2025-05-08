#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

__global__ void upper_triangular_matmul_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             const int N) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Calculate total number of elements in upper triangle
    const int total_elements = (N * (N + 1)) / 2;
    
    for (int idx = tid; idx < total_elements; idx += stride) {
        // Convert linear index to row,col coordinates for upper triangle
        int row = 0;
        int col = 0;
        
        // Find row and column from linear index
        int temp = idx;
        row = (int)(-0.5f + sqrtf(0.25f + 2.0f * temp));
        col = temp - (row * (row + 1)) / 2 + row;
        
        if (row < N && col < N && row <= col) {
            float sum = 0.0f;
            
            // Process 4 elements at a time when possible
            int k = row;
            for (; k <= col - 3; k += 4) {
                float4 a_val = *reinterpret_cast<const float4*>(&A[row * N + k]);
            if (k + 3 <= col) {
                sum += a_val.x * __ldg(&B[k * N + col]);
                sum += a_val.y * __ldg(&B[(k + 1) * N + col]);
                sum += a_val.z * __ldg(&B[(k + 2) * N + col]);
                sum += a_val.w * __ldg(&B[(k + 3) * N + col]);
            }
            
            // Handle remaining elements
            for (; k <= col; k++) {
                sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
            }
            
            C[row * N + col] = sum;
        }
    }
}

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    const int threads = 256;
    const int blocks = (N * N + threads - 1) / threads;
    
    upper_triangular_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Optimized strided upper triangular matrix multiplication");
}