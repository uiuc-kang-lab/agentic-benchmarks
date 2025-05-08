#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define TILE_SIZE 64  // Increased tile size for better memory utilization
#define BLOCK_SIZE 16 // Sub-block size for double buffering

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static cublasHandle_t handle = nullptr;

__global__ void shared_mem_matmul_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        const int M, const int N, const int K) {
    // Shared memory declaration with double buffering
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 2];  // +2 to avoid bank conflicts
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx;

    // Thread block computes sub-matrix
    float sum = 0.0f;
    float reg_a[BLOCK_SIZE];
    float reg_b[BLOCK_SIZE];

    // Initialize registers
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++) {
        reg_a[i] = 0.0f;
        reg_b[i] = 0.0f;
    }

    int buffer = 0;
    
    // Load first tiles into shared memory
    if (row < M && tx < K) {
        As[0][ty][tx] = A[row * K + tx];
    }
    if (col < N && ty < K) {
        Bs[0][ty][tx] = B[ty * N + col];
    }
    __syncthreads();

    // Main loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load next tile while computing current tile
        if (tile + 1 < (K + TILE_SIZE - 1) / TILE_SIZE) {
            int next_tile = (tile + 1) * TILE_SIZE;
            if (row < M && next_tile + tx < K) {
                As[1 - buffer][ty][tx] = A[row * K + next_tile + tx];
            }
            if (col < N && next_tile + ty < K) {
                Bs[1 - buffer][ty][tx] = B[(next_tile + ty) * N + col];
            }
        }

        // Compute using current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k += BLOCK_SIZE) {
            // Load data into registers
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; i++) {
                if (k + i < TILE_SIZE) {
                    reg_a[i] = As[buffer][ty][k + i];
                    reg_b[i] = Bs[buffer][k + i][tx];
                }
            }

            // Compute partial products
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; i++) {
                sum += reg_a[i] * reg_b[i];
            }
        }

        buffer = 1 - buffer;  // Switch buffers
        __syncthreads();
    }

    // Write result
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

    if (M <= 256 && N <= 256 && K <= 256) {
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                      (M + TILE_SIZE - 1) / TILE_SIZE);

        shared_mem_matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
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
    m.def("forward", &forward, "Shared memory optimized matrix multiplication (CUDA)");
}