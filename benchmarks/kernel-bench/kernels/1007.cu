#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define VECTOR_SIZE 4  // float4 loads
#define WARP_SIZE 32

__device__ inline float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K,
    const int lda, const int ldb, const int ldc,
    const bool transA, const bool transB) {

    // Calculate global thread index
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate output indices
    const int total_elements = M * N;
    
    // Each thread processes multiple elements for better utilization
    for (int idx = tid; idx < total_elements; idx += gridDim.x * blockDim.x) {
        const int row = idx / N;
        const int col = idx % N;
        
        if (row < M && col < N) {
            float sum = 0.0f;
            
            // Process elements in chunks of 4 using unrolled scalar loads to avoid misaligned accesses
            int k = 0;
            for (; k + 3 < K; k += 4) {
                float a0 = transA ? A[k * lda + row]       : A[row * lda + k];
                float b0 = transB ? B[col * ldb + k]       : B[k * ldb + col];
                float a1 = transA ? A[(k + 1) * lda + row]   : A[row * lda + k + 1];
                float b1 = transB ? B[col * ldb + (k + 1)]    : B[(k + 1) * ldb + col];
                float a2 = transA ? A[(k + 2) * lda + row]   : A[row * lda + k + 2];
                float b2 = transB ? B[col * ldb + (k + 2)]    : B[(k + 2) * ldb + col];
                float a3 = transA ? A[(k + 3) * lda + row]   : A[row * lda + k + 3];
                float b3 = transB ? B[col * ldb + (k + 3)]    : B[(k + 3) * ldb + col];
                sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
            }
            
            // Handle remaining elements
            for (; k < K; k++) {
                float a_val = transA ? A[k * lda + row] : A[row * lda + k];
                float b_val = transB ? B[col * ldb + k] : B[k * ldb + col];
                sum += a_val * b_val;
            }
            
            C[row * ldc + col] = sum;
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }

    int64_t A_rows = A.size(0);
    int64_t A_cols = A.size(1);
    int64_t B_rows = B.size(0);
    int64_t B_cols = B.size(1);

    bool transA = false;
    bool transB = false;
    int M, N, K;
    int lda, ldb, ldc;

    if (A_rows >= A_cols && B_rows == A_cols) {
        M = A_rows;
        K = A_cols;
        N = B_cols;
        lda = A.stride(0);
        ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        transA = true;
        M = A_cols;
        K = A_rows;
        N = B_cols;
        lda = A.stride(1);
        ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        transB = true;
        M = A_rows;
        K = A_cols;
        N = B_rows;
        lda = A.stride(0);
        ldb = B.stride(1);
    } else if (A_cols > A_rows && B_cols == A_rows) {
        transA = true;
        transB = true;
        M = A_cols;
        K = A_rows;
        N = B_rows;
        lda = A.stride(1);
        ldb = B.stride(1);
    } else {
        throw std::invalid_argument("Incompatible matrix dimensions");
    }

    ldc = N;
    auto C = torch::empty({M, N}, A.options());

    // Calculate optimal grid size based on problem size
    const int total_elements = M * N;
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int max_blocks = 65535;  // Maximum blocks for compute capability
    const int grid_size = min(num_blocks, max_blocks);

    matmul_kernel<<<grid_size, BLOCK_SIZE>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Vectorized 1D Matrix Multiplication (CUDA)");
}