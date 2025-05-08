#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define TILE_SIZE 32
#define VECTOR_SIZE 4
typedef float4 vec_type;

__global__ void optimized_matrix_mult_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            const int M, const int N, const int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Vectorized load for A
        if (row < M && t*TILE_SIZE + tx*VECTOR_SIZE < K) {
            vec_type a_vec = *reinterpret_cast<const vec_type*>(&A[row*K + t*TILE_SIZE + tx*VECTOR_SIZE]);
            As[ty][tx*VECTOR_SIZE+0] = a_vec.x;
            As[ty][tx*VECTOR_SIZE+1] = a_vec.y;
            As[ty][tx*VECTOR_SIZE+2] = a_vec.z;
            As[ty][tx*VECTOR_SIZE+3] = a_vec.w;
        } else {
            for (int v = 0; v < VECTOR_SIZE; v++) {
                As[ty][tx*VECTOR_SIZE+v] = 0.0f;
            }
        }

        // Transposed vector load for B
        if (col < N && t*TILE_SIZE + ty*VECTOR_SIZE < K) {
            vec_type b_vec = *reinterpret_cast<const vec_type*>(&B[(t*TILE_SIZE + ty*VECTOR_SIZE)*N + col]);
            Bs[ty*VECTOR_SIZE+0][tx] = b_vec.x;
            Bs[ty*VECTOR_SIZE+1][tx] = b_vec.y;
            Bs[ty*VECTOR_SIZE+2][tx] = b_vec.z;
            Bs[ty*VECTOR_SIZE+3][tx] = b_vec.w;
        } else {
            for (int v = 0; v < VECTOR_SIZE; v++) {
                Bs[ty*VECTOR_SIZE+v][tx] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row*N + col] = sum;
    }
}

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    dim3 threads(TILE_SIZE/VECTOR_SIZE, TILE_SIZE/VECTOR_SIZE);
    dim3 grid((N + TILE_SIZE-1)/TILE_SIZE, (M + TILE_SIZE-1)/TILE_SIZE);

    optimized_matrix_mult_kernel<<<grid, threads>>>(A.data_ptr<float>(),
                                                  B.data_ptr<float>(),
                                                  C.data_ptr<float>(),
                                                  M, N, K);
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
    m.def("forward", &forward, "Vectorized Matrix Multiplication with Shared Memory Transpose (CUDA)");
}