#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define MAX_BLOCKS 65535

__device__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    if (transpose)
        return matrix[col * ld + row];
    else
        return matrix[row * ld + col];
}

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    bool transA, bool transB) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_threads = blockDim.x * gridDim.x;
    const int thread_id = bid * blockDim.x + tid;
    
    // Calculate number of elements per thread
    const int total_elements = M * N;
    const int elements_per_thread = (total_elements + num_threads - 1) / num_threads;
    
    // Starting point for this thread
    int start_idx = thread_id * elements_per_thread;
    const int end_idx = min(start_idx + elements_per_thread, total_elements);
    
    // Process multiple elements per thread in a strided fashion
    while (start_idx < end_idx) {
        const int row = start_idx / N;
        const int col = start_idx % N;
        
        float sum = 0.0f;
        
        // Use registers for temporary storage
        #pragma unroll 4
        for (int k = 0; k < K; k++) {
            const float a_val = get_element(A, row, k, lda, transA);
            const float b_val = get_element(B, k, col, ldb, transB);
            sum = __fmaf_rn(a_val, b_val, sum);
        }
        
        C[row * ldc + col] = sum;
        start_idx++;
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
    int64_t M, N, K;
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
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
    }

    ldc = N;

    auto C = torch::empty({M, N}, A.options());

    // Calculate optimal grid size based on problem size
    const int total_elements = M * N;
    const int num_blocks = min(MAX_BLOCKS, (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel<<<num_blocks, BLOCK_SIZE>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with tall-and-skinny optimization (CUDA)");
}