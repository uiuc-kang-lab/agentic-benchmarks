#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void triangular_mm_stride_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for(int k = tid; k < N*(N+1)/2; k += stride) {
        float fk = static_cast<float>(k);
        int row = static_cast<int>((sqrtf(8.0f * fk + 1.0f) - 1.0f) * 0.5f);
        
        while((row + 1) * (row + 2)/2 <= k) row++;
        while(row * (row + 1)/2 > k) row--;
        
        int col = k - (row * (row + 1))/2;
        
        float sum = 0.0f;
        for(int m = col; m <= row; m++) {
            sum += A[row * N + m] * B[m * N + col];
        }
        C[row * N + col] = sum;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    int T = N * (N + 1) / 2;
    int block_size = 256;
    int grid_size = (T + block_size - 1) / block_size;

    triangular_mm_stride_kernel<<<grid_size, block_size>>>(A.data_ptr<float>(),
                                                          B.data_ptr<float>(),
                                                          C.data_ptr<float>(),
                                                          N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stride-optimized triangular matrix multiplication (CUDA)");
}