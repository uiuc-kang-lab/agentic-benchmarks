#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    // Calculate total work items in lower triangular part
    const int total_work = (N * (N + 1)) / 2;
    const int num_threads = blockDim.x * gridDim.x;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes multiple elements
    const int items_per_thread = (total_work + num_threads - 1) / num_threads;
    const int start_item = thread_id * items_per_thread;
    const int end_item = min(start_item + items_per_thread, total_work);
    
    // Process assigned work items
    for (int work_idx = start_item; work_idx < end_item; work_idx++) {
        // Convert work index to matrix coordinates
        int row = 0;
        int col = 0;
        
        // Find row and column from work index using quadratic formula
        // k * (k + 1) / 2 <= work_idx < (k + 1) * (k + 2) / 2
        row = static_cast<int>((-1.0f + sqrt(1.0f + 8.0f * work_idx)) / 2.0f);
        col = work_idx - (row * (row + 1)) / 2;
        
        if (row < N && col <= row) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int k = col; k <= row; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    
    // Handle upper triangular part (zeros)
    for (int idx = thread_id; idx < N * N; idx += num_threads) {
        int row = idx / N;
        int col = idx % N;
        if (row < col) {
            C[idx] = 0.0f;
        }
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

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    // Optimize block and grid size based on matrix size
    const int threadsPerBlock = 256;
    const int numBlocks = min(8192, (N * N + threadsPerBlock - 1) / threadsPerBlock);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}