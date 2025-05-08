#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define BLOCK_SIZE 16
// Maximum number of floats that can be stored in constant memory
// (Typically 64KB, i.e., 16384 floats)
#define MAX_CONST_B_ELEMS 16384

// Declare constant memory for matrix B
__constant__ float B_const[MAX_CONST_B_ELEMS];

// Device function to access elements from global memory for matrix A
__device__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    if (transpose)
        return matrix[col * ld + row];
    else
        return matrix[row * ld + col];
}

// Device function to access elements from constant memory for matrix B_const
__device__ float get_const_element(int row, int col, int ld, bool transpose) {
    if (transpose)
        return B_const[col * ld + row];
    else
        return B_const[row * ld + col];
}

// New kernel that uses constant memory for matrix B
__global__ void matmul_kernel_constB(const float* __restrict__ A,
                                       float* __restrict__ C,
                                       int M, int N, int K,
                                       int lda, int ldb_const, int ldc,
                                       bool transA, bool transB) {
    // Compute block and thread indices
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int row = block_row * BLOCK_SIZE + thread_row;
    int col = block_col * BLOCK_SIZE + thread_col;

    float C_value = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        int a_col = t * BLOCK_SIZE + thread_col;
        int b_row = t * BLOCK_SIZE + thread_row;

        // Load tile from matrix A from global memory
        if (row < M && a_col < K)
            As[thread_row][thread_col] = get_element(A, row, a_col, lda, transA);
        else
            As[thread_row][thread_col] = 0.0f;

        // Load tile from constant memory for matrix B
        if (col < N && b_row < K)
            Bs[thread_row][thread_col] = get_const_element(b_row, col, ldb_const, transB);
        else
            Bs[thread_row][thread_col] = 0.0f;

        __syncthreads();

        // Multiply the two tiles
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            C_value += As[thread_row][i] * Bs[i][thread_col];
        }

        __syncthreads();
    }

    // Write the computed value to the output matrix
    if (row < M && col < N)
        C[row * ldc + col] = C_value;
}

// Host interface function using pybind11
// Assumes that matrix B fits in constant memory (i.e., K*N <= MAX_CONST_B_ELEMS).
// The kernel computes C = A * B with possible transpositions based on input shapes.

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure the input tensors are CUDA tensors and 2D matrices
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }

    // Retrieve dimensions
    int64_t A_rows = A.size(0);
    int64_t A_cols = A.size(1);
    int64_t B_rows = B.size(0);
    int64_t B_cols = B.size(1);

    bool transA = false;
    bool transB = false;
    int64_t M, N, K;
    int lda, ldb_const, ldc;

    // Determine correct multiplication case based on input shapes
    if (A_rows >= A_cols && B_rows == A_cols) {
        // A (M x K), B (K x N)
        M = A_rows;
        K = A_cols;
        N = B_cols;
        lda = A.stride(0);
        ldb_const = B.stride(0);  // For contiguous B: stride(0) equals number of columns (N)
    } else if (A_cols > A_rows && B_rows == A_rows) {
        // A (K x M), needs to be transposed
        transA = true;
        M = A_cols;
        K = A_rows;
        N = B_cols;
        lda = A.stride(1);
        ldb_const = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        // B is transposed: B (N x K)
        transB = true;
        M = A_rows;
        K = A_cols;
        N = B_rows;
        lda = A.stride(0);
        ldb_const = B.stride(1);  // For transposed B, use stride(1)
    } else if (A_cols > A_rows && B_cols == A_rows) {
        // Both A and B require transposition
        transA = true;
        transB = true;
        M = A_cols;
        K = A_rows;
        N = B_rows;
        lda = A.stride(1);
        ldb_const = B.stride(1);
    } else {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
    }
    
    ldc = N;

    // Check that matrix B fits into constant memory
    if (K * N > MAX_CONST_B_ELEMS) {
        std::cerr << "Matrix B is too large to fit in constant memory: " << K << " x " << N << std::endl;
        throw std::invalid_argument("Matrix B is too large to fit in constant memory");
    }

    // Ensure matrix B is contiguous
    auto B_contig = B.contiguous();

    // Copy B into constant memory
    cudaMemcpyToSymbol(B_const, B_contig.data_ptr<float>(), sizeof(float) * K * N);

    // Allocate output tensor C
    auto C = torch::empty({M, N}, A.options());

    // Configure grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel
    matmul_kernel_constB<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb_const, ldc,
        transA, transB
    );

    cudaDeviceSynchronize();
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with tall-and-skinny optimization using constant memory (CUDA)");
}
