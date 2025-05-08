#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// Constant memory for frequently accessed parameters
__constant__ int const_dims[6];  // M, N, K, lda, ldb, ldc
__constant__ bool const_trans[2]; // transA, transB

__device__ __forceinline__ float get_element(const float* __restrict__ matrix, int row, int col) {
    const int ld = (matrix == nullptr) ? const_dims[3] : const_dims[4];
    const bool trans = (matrix == nullptr) ? const_trans[0] : const_trans[1];
    return trans ? matrix[col * ld + row] : matrix[row * ld + col];
}

__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    int row = block_row * BLOCK_SIZE + thread_row;
    int col = block_col * BLOCK_SIZE + thread_col;

    float C_value = 0.0f;

    for (int t = 0; t < (const_dims[2] + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        if (row < const_dims[0] && t * BLOCK_SIZE + thread_col < const_dims[2]) {
            As[thread_row][thread_col] = get_element(A, row, t * BLOCK_SIZE + thread_col);
        } else {
            As[thread_row][thread_col] = 0.0f;
        }

        if (col < const_dims[1] && t * BLOCK_SIZE + thread_row < const_dims[2]) {
            Bs[thread_row][thread_col] = get_element(B, t * BLOCK_SIZE + thread_row, col);
        } else {
            Bs[thread_row][thread_col] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            C_value += As[thread_row][k] * Bs[k][thread_col];
        }

        __syncthreads();
    }

    if (row < const_dims[0] && col < const_dims[1]) {
        C[row * const_dims[5] + col] = C_value;
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
    int dims[6];  // M, N, K, lda, ldb, ldc
    bool trans[2] = {false, false};

    if (A_rows >= A_cols && B_rows == A_cols) {
        dims[0] = A_rows;
        dims[2] = A_cols;
        dims[1] = B_cols;
        dims[3] = A.stride(0);
        dims[4] = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        trans[0] = true;
        dims[0] = A_cols;
        dims[2] = A_rows;
        dims[1] = B_cols;
        dims[3] = A.stride(1);
        dims[4] = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        trans[1] = true;
        dims[0] = A_rows;
        dims[2] = A_cols;
        dims[1] = B_rows;
        dims[3] = A.stride(0);
        dims[4] = B.stride(1);
    } else if (A_cols > A_rows && B_cols == A_rows) {
        trans[0] = trans[1] = true;
        dims[0] = A_cols;
        dims[2] = A_rows;
        dims[1] = B_rows;
        dims[3] = A.stride(1);
        dims[4] = B.stride(1);
    } else {
        throw std::invalid_argument("Incompatible matrix dimensions");
    }

    dims[5] = dims[1];  // ldc = N

    // Copy constants to constant memory
    cudaMemcpyToSymbol(const_dims, dims, sizeof(dims));
    cudaMemcpyToSymbol(const_trans, trans, sizeof(trans));

    auto C = torch::empty({dims[0], dims[1]}, A.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((dims[1] + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (dims[0] + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with constant memory optimization (CUDA)");
}