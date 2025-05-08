#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 32
#define VECTOR_SIZE 4  // Use vector loads for better memory coalescing

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static cublasHandle_t handle = nullptr;

// Vectorized load type for better memory coalescing
typedef float4 vector_t;

__global__ void coalesced_matmul_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       const int M, const int N, const int K) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Compute base indices for coalesced memory access
    const int row = by * BLOCK_SIZE + ty;
    const int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles with vectorized loads
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // Compute aligned addresses for vectorized loads
        const int baseIdxA = row * K + tile * BLOCK_SIZE;
        const int baseIdxB = (tile * BLOCK_SIZE) * N + col;
        
        // Load A tile with vectorized reads where possible
        if (row < M && (tile * BLOCK_SIZE + tx) < K) {
            if ((baseIdxA + tx) % VECTOR_SIZE == 0 && tx + VECTOR_SIZE <= BLOCK_SIZE) {
                vector_t v = *reinterpret_cast<const vector_t*>(&A[baseIdxA + tx]);
                As[ty][tx] = v.x;
                if (tx + 1 < BLOCK_SIZE) As[ty][tx + 1] = v.y;
                if (tx + 2 < BLOCK_SIZE) As[ty][tx + 2] = v.z;
                if (tx + 3 < BLOCK_SIZE) As[ty][tx + 3] = v.w;
            } else {
                As[ty][tx] = A[baseIdxA + tx];
            }
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile with vectorized reads where possible
        if ((tile * BLOCK_SIZE + ty) < K && col < N) {
            if ((baseIdxB + ty * N) % VECTOR_SIZE == 0) {
                vector_t v = *reinterpret_cast<const vector_t*>(&B[baseIdxB + ty * N]);
                Bs[ty][tx] = v.x;
            } else {
                Bs[ty][tx] = B[baseIdxB + ty * N];
            }
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result with coalesced access
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    if (M <= 128 && N <= 128 && K <= 128) {
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

        coalesced_matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    } else {
        if (handle == nullptr) {
            cublasCreate(&handle);
            cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int M = A.size(0);
    const int N = B.size(1);

    auto options = torch::TensorOptions()
                      .dtype(A.dtype())
                      .device(A.device())
                      .requires_grad(false);
    
    torch::Tensor C = torch::empty({M, N}, options);
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced matrix multiplication (CUDA)");
}