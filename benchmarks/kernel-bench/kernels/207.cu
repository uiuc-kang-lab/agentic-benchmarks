#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define TILE_SIZE 32
#define THREADS_PER_BLOCK 16

__global__ void optimized_mm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty * 2;
    int col = bx * TILE_SIZE + tx * 2;
    
    float sum[2][2] = {0.0f};

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < M && (t * TILE_SIZE + tx) < K) {
            As[ty * 2][tx] = A[row * K + t * TILE_SIZE + tx];
            As[ty * 2 + 1][tx] = A[(row + 1) * K + t * TILE_SIZE + tx];
        } else {
            As[ty * 2][tx] = 0.0f;
            As[ty * 2 + 1][tx] = 0.0f;
        }

        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && (bx * TILE_SIZE + tx) < N) {
            Bs[ty][tx * 2] = B[b_row * N + bx * TILE_SIZE + tx * 2];
            Bs[ty][tx * 2 + 1] = B[b_row * N + bx * TILE_SIZE + tx * 2 + 1];
        } else {
            Bs[ty][tx * 2] = 0.0f;
            Bs[ty][tx * 2 + 1] = 0.0f;
        }
        
        __syncthreads();

        // Compute 2x2 elements per thread
        for (int k = 0; k < TILE_SIZE; ++k) {
            float a1 = As[ty * 2][k];
            float a2 = As[ty * 2 + 1][k];
            float b1 = Bs[k][tx * 2];
            float b2 = Bs[k][tx * 2 + 1];
            
            sum[0][0] += a1 * b1;
            sum[0][1] += a1 * b2;
            sum[1][0] += a2 * b1;
            sum[1][1] += a2 * b2;
        }
        __syncthreads();
    }

    // Write 2x2 results
    if (row < M && col < N) {
        C[row * N + col] = sum[0][0];
        if (col + 1 < N) C[row * N + col + 1] = sum[0][1];
        if (row + 1 < M && col < N) C[(row + 1) * N + col] = sum[1][0];
        if (row + 1 < M && col + 1 < N) C[(row + 1) * N + col + 1] = sum[1][1];
    }
}

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const float *d_A = A.data_ptr<float>();
    const float *d_B = B.data_ptr<float>();
    float *d_C = C.data_ptr<float>();

    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    optimized_mm_kernel<<<grid, threads>>>(d_A, d_B, d_C, M, N, K);
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int N = B.size(1);

    torch::Tensor C = torch::empty({M, N}, A.options());
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized matrix multiplication with thread coarsening (CUDA)");
}
