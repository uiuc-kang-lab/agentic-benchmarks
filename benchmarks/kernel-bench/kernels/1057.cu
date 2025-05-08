#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_X 32
#define BLOCK_Y 8

__device__ float get_element(const float* matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    bool transA, bool transB) {

    __shared__ float As[BLOCK_Y][BLOCK_X];
    __shared__ float Bs[BLOCK_X][BLOCK_Y]; // Transposed storage

    int row = blockIdx.y * BLOCK_Y + threadIdx.y;
    int col = blockIdx.x * BLOCK_X + threadIdx.x;
    
    float acc = 0.0f;

    for (int t = 0; t < (K + BLOCK_X - 1) / BLOCK_X; ++t) {
        // Coalesced A load: Threads in same block read consecutive A elements
        int a_col = t * BLOCK_X + threadIdx.x;
        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = get_element(A, row, a_col, lda, transA);
        else
            As[threadIdx.y][threadIdx.x] = 0;

        // Coalesced B load with implicit transpose
        int b_row = t * BLOCK_X + threadIdx.x;
        if (col < N && b_row < K)
            Bs[threadIdx.y][threadIdx.x] = get_element(B, b_row, col, ldb, transB);
        else
            Bs[threadIdx.x][threadIdx.y] = 0;

        __syncthreads();

        // Accumulate products with coalesced shared memory access
        for (int k = 0; k < BLOCK_X; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.y];
        }
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * ldc + col] = acc;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda()) throw std::invalid_argument("Inputs must be CUDA tensors");
    
    int64_t M, N, K;
    bool transA = false, transB = false;
    int lda, ldb, ldc;

    // Original dimension and transpose logic (kept intact)
    int64_t A_rows = A.size(0), A_cols = A.size(1);
    int64_t B_rows = B.size(0), B_cols = B.size(1);

    if (A_rows >= A_cols && B_rows == A_cols) {
        M = A_rows; K = A_cols; N = B_cols;
        lda = A.stride(0); ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        transA = true; M = A_cols; K = A_rows; N = B_cols;
        lda = A.stride(1); ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        transB = true; M = A_rows; K = A_cols; N = B_rows;
        lda = A.stride(0); ldb = B.stride(1);
    } else if (A_cols > A_rows && B_cols == A_rows) {
        transA = transB = true; M = A_cols; K = A_rows; N = B_rows;
        lda = A.stride(1); ldb = B.stride(1);
    } else throw std::invalid_argument("Incompatible dimensions");

    ldc = N;
    auto C = torch::empty({M, N}, A.options());

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(
        (N + BLOCK_X - 1) / BLOCK_X,
        (M + BLOCK_Y - 1) / BLOCK_Y
    );

    matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB
    );

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Coalesced-access optimized matrix multiplication");
}