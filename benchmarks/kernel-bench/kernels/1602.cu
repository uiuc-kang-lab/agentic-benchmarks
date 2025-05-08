#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void upper_triangular_matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = threadIdx.x % 32;
    
    if (row < N && col < N && row <= col) {
        float sum = 0.0f;
        
        // Each thread processes multiple elements within its row-col assignment
        for (int k = row + lane_id; k <= col; k += 32) {
            if (k <= col) {
                sum += A[row * N + k] * B[k * N + col];
            }
        }
        
        // Perform warp-level reduction
        sum = warp_reduce_sum(sum);
        
        // Only the first thread in the warp writes the result
        if (lane_id == 0) {
            C[row * N + col] = sum;
        }
    }
}

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    // Adjust block size to ensure proper warp alignment
    dim3 threadsPerBlock(32, 16);  // 32 threads per warp
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    upper_triangular_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Upper triangular matrix multiplication");
}